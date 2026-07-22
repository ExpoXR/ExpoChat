import json
import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager
from datetime import UTC, datetime
from typing import Any

from .config import settings
from .migrations import apply_migrations

MAX_RUN_EVENTS = 500


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


def add_event(run_id: str, event_type: str, message: str, data: Any = None) -> int:
    with closing(connect()) as db:
        cursor = db.execute(
            "insert into run_events(run_id,event_type,message,data_json,created_at) values(?,?,?,?,?)",
            (run_id, event_type, message, json.dumps(data, ensure_ascii=False) if data is not None else None, utcnow()),
        )
        db.execute(
            "delete from run_events where run_id=? and id not in "
            "(select id from run_events where run_id=? order by id desc limit ?)",
            (run_id, run_id, MAX_RUN_EVENTS),
        )
        db.commit()
        return int(cursor.lastrowid)


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
              provider text primary key check(provider in ('codex','claude')),
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
