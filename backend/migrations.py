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


MIGRATIONS: list[tuple[int, Migration]] = [
    (1, _snapshot_metadata),
    (2, _durable_jobs),
    (3, _run_usage),
    (4, _unique_active_jobs),
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
