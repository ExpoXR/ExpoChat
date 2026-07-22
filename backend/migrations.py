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


MIGRATIONS: list[tuple[int, Migration]] = [
    (1, _snapshot_metadata),
    (2, _durable_jobs),
    (3, _run_usage),
    (4, _unique_active_jobs),
    (5, _chat_pinned),
    (6, _brain_gemini),
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
