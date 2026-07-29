import hashlib
import sqlite3
from collections.abc import Callable

Migration = Callable[[sqlite3.Connection], None]


def _columns(db: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in db.execute(f"pragma table_info({table})")}


def _add_columns(db: sqlite3.Connection, table: str, definitions: dict[str, str]) -> None:
    existing = _columns(db, table)
    for name, definition in definitions.items():
        if name not in existing:
            db.execute(f"alter table {table} add column {name} {definition}")


def _snapshot_metadata(db: sqlite3.Connection) -> None:
    _add_columns(
        db,
        "snapshots",
        {
            "archive_deleted_at": "text",
            "source_bytes": "integer not null default 0",
            "archive_bytes": "integer not null default 0",
            "status": "text not null default 'ready'",
        },
    )


def _durable_jobs(db: sqlite3.Connection) -> None:
    _add_columns(
        db,
        "jobs",
        {
            "started_at": "text",
            "completed_at": "text",
            "lease_owner": "text",
            "lease_expires_at": "text",
            "cancel_requested_at": "text",
        },
    )
    db.execute("create index if not exists idx_jobs_run_type_status on jobs(run_id,job_type,status)")


def _run_usage(db: sqlite3.Connection) -> None:
    _add_columns(db, "runs", {"usage_json": "text not null default '{}'"})


def _chat_pinned(db: sqlite3.Connection) -> None:
    _add_columns(db, "chats", {"pinned": "integer not null default 0"})


def _unique_active_jobs(db: sqlite3.Connection) -> None:
    db.execute(
        "update jobs set status='cancelled',error='Duplicate active job removed during migration' "
        "where status in ('pending','running') and exists ("
        "select 1 from jobs earlier where earlier.run_id=jobs.run_id and earlier.job_type=jobs.job_type "
        "and earlier.status in ('pending','running') and earlier.id<jobs.id)"
    )
    db.execute(
        "create unique index if not exists idx_jobs_one_active_type "
        "on jobs(run_id,job_type) where status in ('pending','running')"
    )


def _brain_gemini(db: sqlite3.Connection) -> None:
    # SQLite can't ALTER a CHECK constraint, so rebuild brain_configs allowing 'gemini'.
    check = db.execute(
        "select sql from sqlite_master where type='table' and name='brain_configs'"
    ).fetchone()
    if check and "gemini" in (check[0] or ""):
        return
    db.execute(
        """
        create table brain_configs_new (
          provider text primary key check(provider in ('codex','claude','gemini')),
          model text not null, key_ciphertext text, source text not null default 'environment',
          enabled integer not null default 0, validated_at text, last_error text,
          updated_at text not null
        )
        """
    )
    db.execute(
        "insert into brain_configs_new(provider,model,key_ciphertext,source,enabled,validated_at,last_error,updated_at) "
        "select provider,model,key_ciphertext,source,enabled,validated_at,last_error,updated_at from brain_configs"
    )
    db.execute("drop table brain_configs")
    db.execute("alter table brain_configs_new rename to brain_configs")


def _settings_and_ledger(db: sqlite3.Connection) -> None:
    db.execute(
        "create table if not exists app_settings ("
        " key text primary key, value text not null, updated_at text not null)"
    )
    db.execute(
        "create table if not exists usage_ledger ("
        " id integer primary key autoincrement, day text not null, source text not null,"
        " provider text not null, input integer not null default 0, output integer not null default 0,"
        " total integer not null default 0, created_at text not null)"
    )
    db.execute("create index if not exists idx_usage_ledger_day on usage_ledger(day)")


def _plan_history(db: sqlite3.Connection) -> None:
    db.execute(
        "create table if not exists plan_versions ("
        " id integer primary key autoincrement, run_id text not null, version integer not null,"
        " kind text not null, content text not null, brain_provider text, created_at text not null,"
        " foreign key(run_id) references runs(id) on delete cascade)"
    )
    db.execute("create index if not exists idx_plan_versions_run on plan_versions(run_id,version)")
    # Persist the Agent-Mode supervisor plan alongside the assistant chat message.
    _add_columns(db, "messages", {"plan": "text"})


def _task_dag(db: sqlite3.Connection) -> None:
    db.execute(
        "create table if not exists subtasks ("
        " id text primary key, run_id text not null, node_id text not null,"
        " title text not null, spec text not null,"
        " depends_on_json text not null default '[]', file_globs_json text not null default '[]',"
        " role text not null default 'implementation', status text not null default 'pending',"
        " agent_id text, worktree_ref text, result_summary text, verdict text,"
        " attempts integer not null default 0, created_at text not null, updated_at text not null,"
        " unique(run_id, node_id), foreign key(run_id) references runs(id) on delete cascade)"
    )
    db.execute("create index if not exists idx_subtasks_run_status on subtasks(run_id,status)")
    db.execute(
        "create table if not exists subtask_results ("
        " id integer primary key autoincrement, subtask_id text not null, run_id text not null,"
        " kind text not null, content text not null, created_at text not null,"
        " foreign key(subtask_id) references subtasks(id) on delete cascade)"
    )
    db.execute("create index if not exists idx_subtask_results_subtask on subtask_results(subtask_id,id)")

    # Extend jobs.job_type (SQLite can't ALTER a CHECK) to allow subtask/merge, add node_id,
    # and re-scope uniqueness so many subtask jobs can be active at once (one per node),
    # while research/implementation/merge stay one-active-per-type.
    current = db.execute("select sql from sqlite_master where type='table' and name='jobs'").fetchone()
    if current and "'subtask'" in (current[0] or ""):
        return
    db.execute(
        "create table jobs_new ("
        " id integer primary key autoincrement, run_id text not null,"
        " job_type text not null check(job_type in ('research','implementation','subtask','merge')),"
        " node_id text, status text not null default 'pending', attempts integer not null default 0,"
        " error text, created_at text not null, updated_at text not null,"
        " started_at text, completed_at text, lease_owner text, lease_expires_at text,"
        " cancel_requested_at text, foreign key(run_id) references runs(id) on delete cascade)"
    )
    db.execute(
        "insert into jobs_new(id,run_id,job_type,status,attempts,error,created_at,updated_at,"
        "started_at,completed_at,lease_owner,lease_expires_at,cancel_requested_at) "
        "select id,run_id,job_type,status,attempts,error,created_at,updated_at,"
        "started_at,completed_at,lease_owner,lease_expires_at,cancel_requested_at from jobs"
    )
    db.execute("drop table jobs")
    db.execute("alter table jobs_new rename to jobs")
    db.execute("create index if not exists idx_jobs_status_id on jobs(status,id)")
    db.execute("create index if not exists idx_jobs_run_type_status on jobs(run_id,job_type,status)")
    db.execute(
        "create unique index if not exists idx_jobs_one_active_type on jobs(run_id,job_type) "
        "where status in ('pending','running') and job_type in ('research','implementation','merge')"
    )
    db.execute(
        "create unique index if not exists idx_jobs_one_active_node on jobs(run_id,node_id) "
        "where status in ('pending','running') and node_id is not null"
    )


def _brain_memory(db: sqlite3.Connection) -> None:
    db.execute(
        "create table if not exists brain_memory ("
        " id integer primary key autoincrement, run_id text not null,"
        " seq integer not null, step text not null, role text not null,"
        " content text not null, tokens_estimate integer not null default 0,"
        " created_at text not null,"
        " unique(run_id, seq), foreign key(run_id) references runs(id) on delete cascade)"
    )
    db.execute("create index if not exists idx_brain_memory_run on brain_memory(run_id, seq)")
    # Add suggested_model to subtasks for per-subtask agent hints.
    _add_columns(db, "subtasks", {"suggested_model": "text"})


def _subtask_acceptance(db: sqlite3.Connection) -> None:
    _add_columns(db, "subtasks", {"acceptance_criteria": "text not null default ''"})


def _check_evidence(db: sqlite3.Connection) -> None:
    db.execute(
        "create table if not exists check_evidence ("
        " id integer primary key autoincrement, run_id text not null,"
        " cycle integer not null, command text not null, args_json text not null default '[]',"
        " exit_code integer not null, output text not null default '',"
        " duration_ms integer not null default 0, workspace_hash text not null default '',"
        " node_id text, created_at text not null,"
        " foreign key(run_id) references runs(id) on delete cascade)"
    )
    db.execute("create index if not exists idx_check_evidence_run on check_evidence(run_id,cycle)")


def _subtask_graph_ui(db: sqlite3.Connection) -> None:
    # Visual task-graph: brain-marked task difficulty + user's per-task agent override.
    _add_columns(
        db,
        "subtasks",
        {"complexity": "text not null default 'simple'", "assigned_agent_id": "text"},
    )


def _offline_safe_graph(db: sqlite3.Connection) -> None:
    _add_columns(
        db,
        "runs",
        {
            "graph_plan_hash": "text",
            "plan_state": "text not null default 'none'",
            "wait_reason": "text",
            "next_retry_at": "text",
            "resume_status": "text",
        },
    )
    _add_columns(
        db,
        "subtasks",
        {
            "handoff_json": "text",
            "blocked_reason": "text",
            "input_manifest_json": "text",
            "output_manifest_json": "text",
            "delta_manifest_json": "text",
        },
    )
    legacy_runs = list(
        db.execute("select id,status,draft_plan,approved_plan from runs where plan_state='none'")
    )
    for run_id, status, draft_plan, approved_plan in legacy_runs:
        plan = draft_plan or approved_plan
        if not plan:
            continue
        has_graph = db.execute("select 1 from subtasks where run_id=? limit 1", (run_id,)).fetchone()
        graph_hash = hashlib.sha256(str(plan).strip().encode()).hexdigest() if has_graph else None
        state = "refined" if status in {
            "plan_ready", "awaiting_approval", "implementing", "verifying", "applying",
            "post_check", "completed", "failed", "rolled_back",
        } else "none"
        db.execute(
            "update runs set plan_state=?,graph_plan_hash=coalesce(graph_plan_hash,?) where id=?",
            (state, graph_hash, run_id),
        )
    current = db.execute("select sql from sqlite_master where type='table' and name='jobs'").fetchone()
    if current and "'provisional'" not in (current[0] or ""):
        db.execute(
            "create table jobs_offline_new ("
            " id integer primary key autoincrement, run_id text not null,"
            " job_type text not null check(job_type in "
            "('provisional','decompose','research','implementation','subtask','merge')),"
            " node_id text, status text not null default 'pending', attempts integer not null default 0,"
            " error text, created_at text not null, updated_at text not null,"
            " started_at text, completed_at text, lease_owner text, lease_expires_at text,"
            " cancel_requested_at text, next_attempt_at text, wait_reason text,"
            " foreign key(run_id) references runs(id) on delete cascade)"
        )
        db.execute(
            "insert into jobs_offline_new(id,run_id,job_type,node_id,status,attempts,error,created_at,updated_at,"
            "started_at,completed_at,lease_owner,lease_expires_at,cancel_requested_at) "
            "select id,run_id,job_type,node_id,status,attempts,error,created_at,updated_at,"
            "started_at,completed_at,lease_owner,lease_expires_at,cancel_requested_at from jobs"
        )
        db.execute("drop table jobs")
        db.execute("alter table jobs_offline_new rename to jobs")
    else:
        _add_columns(db, "jobs", {"next_attempt_at": "text", "wait_reason": "text"})
    db.execute("create index if not exists idx_jobs_status_id on jobs(status,id)")
    db.execute("create index if not exists idx_jobs_run_type_status on jobs(run_id,job_type,status)")
    db.execute(
        "create unique index if not exists idx_jobs_one_active_type on jobs(run_id,job_type) "
        "where status in ('pending','running','waiting_ollama') "
        "and job_type in ('provisional','decompose','research','implementation','merge')"
    )
    db.execute(
        "create unique index if not exists idx_jobs_one_active_node on jobs(run_id,node_id) "
        "where status in ('pending','running','waiting_ollama') and node_id is not null"
    )


MIGRATIONS: list[tuple[int, Migration]] = [
    (1, _snapshot_metadata),
    (2, _durable_jobs),
    (3, _run_usage),
    (4, _unique_active_jobs),
    (5, _chat_pinned),
    (6, _brain_gemini),
    (7, _settings_and_ledger),
    (8, _plan_history),
    (9, _task_dag),
    (10, _brain_memory),
    (11, _subtask_acceptance),
    (12, _check_evidence),
    (13, _subtask_graph_ui),
    (14, _offline_safe_graph),
]


def apply_migrations(db: sqlite3.Connection) -> None:
    db.execute(
        "create table if not exists schema_migrations (version integer primary key, applied_at text not null)"
    )
    applied = {int(row[0]) for row in db.execute("select version from schema_migrations")}
    for version, migration in MIGRATIONS:
        if version in applied:
            continue
        migration(db)
        db.execute(
            "insert into schema_migrations(version,applied_at) values(?,strftime('%Y-%m-%dT%H:%M:%fZ','now'))",
            (version,),
        )
