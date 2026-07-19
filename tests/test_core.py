import json
import os
import sqlite3
import sys
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

TEST_ROOT = Path(os.environ["OLLMA_TEST_ROOT"])
WORKSPACES = TEST_ROOT / "workspaces"

from fastapi import HTTPException  # noqa: E402

from backend import brain_runner, db, orchestrator, workspace  # noqa: E402
from backend.main import pinned_context  # noqa: E402
from backend.run_state import validate_transition  # noqa: E402
from backend.security import decrypt_secret, encrypt_secret  # noqa: E402
from backend.workspace import (  # noqa: E402
    allowed_path,
    create_snapshot,
    manifest_hash,
    restore_snapshot,
    stage_workspace,
)
from backend.workspace_tools import run_check, search_text  # noqa: E402


def setup_module():
    db.init_db()


def clear_workflow_tables():
    for table in (
        "jobs", "verification_results", "run_approvals", "history_snippets",
        "run_artifacts", "run_events", "runs", "agent_profiles", "snapshots",
    ):
        db.execute(f"delete from {table}")


def seed_agent(agent_id: str, model: str, roles: list[str], capabilities: list[str], priority: int = 50):
    now = db.utcnow()
    db.execute(
        "insert into agent_profiles(id,name,model,roles_json,capabilities_json,context_size,priority,role_scores_json,discovered_at,updated_at) values(?,?,?,?,?,?,?,?,?,?)",
        (agent_id, agent_id, model, json.dumps(roles), json.dumps(capabilities), 32768, priority, json.dumps({role: 80 for role in roles}), now, now),
    )


def test_sqlite_wal_and_foreign_keys():
    with db.connect() as conn:
        assert conn.execute("pragma foreign_keys").fetchone()[0] == 1
        assert conn.execute("pragma journal_mode").fetchone()[0].lower() == "wal"


def test_migrations_are_idempotent():
    db.init_db()
    db.init_db()
    versions = db.all_rows("select version from schema_migrations order by version")
    assert [row["version"] for row in versions] == [1, 2, 3, 4]


def test_credential_round_trip():
    encrypted = encrypt_secret("secret-value")
    assert "secret-value" not in encrypted
    assert decrypt_secret(encrypted) == "secret-value"


def test_codex_runner_applies_api_key_before_starting_thread(monkeypatch):
    calls = []

    class FakeThread:
        def run(self, prompt):
            calls.append(("run", prompt))
            return SimpleNamespace(final_response="OK")

    class FakeCodex:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def login_api_key(self, api_key):
            calls.append(("login", api_key))

        def thread_start(self, *, model, sandbox):
            calls.append(("start", model, sandbox))
            return FakeThread()

    fake_module = SimpleNamespace(Codex=FakeCodex, Sandbox=SimpleNamespace(read_only="read-only"))
    monkeypatch.setitem(sys.modules, "openai_codex", fake_module)

    result = brain_runner.run_codex({"api_key": "secret", "model": "gpt-test", "prompt": "ping"})

    assert result == "OK"
    assert calls == [
        ("login", "secret"),
        ("start", "gpt-test", "read-only"),
        ("run", "ping"),
    ]


def test_allowed_path_rejects_symlink_escape():
    outside = TEST_ROOT / "outside"
    outside.mkdir(exist_ok=True)
    link = WORKSPACES / "escape"
    link.unlink(missing_ok=True)
    link.symlink_to(outside, target_is_directory=True)
    try:
        allowed_path(link)
        assert False, "symlink escape accepted"
    except HTTPException as exc:
        assert exc.status_code == 403


def test_stage_ignores_symlinks_and_dependencies():
    source = WORKSPACES / "stage-source"
    source.mkdir(exist_ok=True)
    (source / "app.py").write_text("print('ok')\n")
    (source / "node_modules").mkdir(exist_ok=True)
    (source / "node_modules" / "x.js").write_text("x")
    link = source / "outside-link"
    link.unlink(missing_ok=True)
    link.symlink_to(TEST_ROOT / "outside", target_is_directory=True)
    stage = stage_workspace("a" * 24, source)
    assert (stage / "app.py").is_file()
    assert not (stage / "node_modules").exists()
    assert not (stage / "outside-link").exists()


def test_snapshot_restore_replaces_workspace():
    clear_workflow_tables()
    target = WORKSPACES / "restore-target"
    target.mkdir(exist_ok=True)
    file = target / "value.txt"
    file.write_text("before")
    snapshot = create_snapshot(target)
    file.write_text("after")
    (target / "new.txt").write_text("remove")
    restore_snapshot(snapshot["id"])
    assert file.read_text() == "before"
    assert not (target / "new.txt").exists()


def test_snapshot_restore_preserves_excluded_directories():
    clear_workflow_tables()
    target = WORKSPACES / "restore-preserved"
    target.mkdir(exist_ok=True)
    (target / "value.txt").write_text("before")
    git_dir = target / ".git"
    dependency_dir = target / "node_modules"
    git_dir.mkdir(exist_ok=True)
    dependency_dir.mkdir(exist_ok=True)
    (git_dir / "HEAD").write_text("current history")
    (dependency_dir / "package.js").write_text("installed dependency")
    snapshot = create_snapshot(target)
    (target / "value.txt").write_text("after")
    (target / "new.txt").write_text("remove")
    (git_dir / "HEAD").write_text("new history")
    restore_snapshot(snapshot["id"])
    assert (target / "value.txt").read_text() == "before"
    assert not (target / "new.txt").exists()
    assert (git_dir / "HEAD").read_text() == "new history"
    assert (dependency_dir / "package.js").read_text() == "installed dependency"


def test_snapshot_records_sizes_and_cleans_failure_artifacts(monkeypatch):
    clear_workflow_tables()
    target = WORKSPACES / "atomic-snapshot"
    target.mkdir(exist_ok=True)
    (target / "value.txt").write_text("snapshot payload")
    created = workspace.create_snapshot(target)
    row = db.one("select * from snapshots where id=?", (created["id"],))
    assert row["source_bytes"] == len("snapshot payload")
    assert row["archive_bytes"] > 0
    workspace.discard_snapshot(created["id"])
    archives_before_failure = set(Path(workspace.settings.snapshot_dir).glob("snap-*.tar.gz"))

    original_execute = workspace.db.execute

    def fail_insert(sql, args=()):
        if sql.startswith("insert into snapshots"):
            raise RuntimeError("database failure")
        return original_execute(sql, args)

    monkeypatch.setattr(workspace.db, "execute", fail_insert)
    with pytest.raises(RuntimeError, match="database failure"):
        workspace.create_snapshot(target)
    assert not list(Path(workspace.settings.snapshot_dir).glob("*.part"))
    assert set(Path(workspace.settings.snapshot_dir).glob("snap-*.tar.gz")) == archives_before_failure


def test_snapshot_size_limit(monkeypatch):
    target = WORKSPACES / "oversized-snapshot"
    target.mkdir(exist_ok=True)
    (target / "large.txt").write_text("too large")
    monkeypatch.setattr(workspace, "settings", replace(workspace.settings, snapshot_max_bytes=1))
    with pytest.raises(RuntimeError, match="exceeds limit"):
        workspace.create_snapshot(target)
    assert not list(Path(workspace.settings.snapshot_dir).glob("*.part"))


def test_snapshot_rejects_low_disk_and_cleans_rename_failure(monkeypatch):
    target = WORKSPACES / "snapshot-faults"
    target.mkdir(exist_ok=True)
    (target / "value.txt").write_text("payload")
    low_space = type("DiskUsage", (), {"free": 0})()
    monkeypatch.setattr(workspace.shutil, "disk_usage", lambda _: low_space)
    with pytest.raises(RuntimeError, match="Insufficient snapshot storage"):
        workspace.create_snapshot(target)
    monkeypatch.undo()

    archives = set(Path(workspace.settings.snapshot_dir).glob("snap-*.tar.gz"))
    monkeypatch.setattr(workspace.os, "replace", lambda *_: (_ for _ in ()).throw(OSError("rename failure")))
    with pytest.raises(OSError, match="rename failure"):
        workspace.create_snapshot(target)
    assert not list(Path(workspace.settings.snapshot_dir).glob("*.part"))
    assert set(Path(workspace.settings.snapshot_dir).glob("snap-*.tar.gz")) == archives


def test_snapshot_cleans_tar_and_verification_failures(monkeypatch):
    target = WORKSPACES / "snapshot-tar-faults"
    target.mkdir(exist_ok=True)
    (target / "value.txt").write_text("payload")
    archives = set(Path(workspace.settings.snapshot_dir).glob("snap-*.tar.gz"))
    original_open = workspace.tarfile.open
    monkeypatch.setattr(
        workspace.tarfile,
        "open",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("tar failure")),
    )
    with pytest.raises(OSError, match="tar failure"):
        workspace.create_snapshot(target)
    assert not list(Path(workspace.settings.snapshot_dir).glob("*.part"))

    monkeypatch.setattr(workspace.tarfile, "open", original_open)
    monkeypatch.setattr(workspace.tarfile.TarFile, "getmembers", lambda _: [])
    with pytest.raises(RuntimeError, match="verification failed"):
        workspace.create_snapshot(target)
    assert not list(Path(workspace.settings.snapshot_dir).glob("*.part"))
    assert set(Path(workspace.settings.snapshot_dir).glob("snap-*.tar.gz")) == archives


def test_orphan_report_requires_explicit_cleanup():
    orphan = Path(workspace.settings.snapshot_dir) / "snap-1000000000-deadbeef.tar.gz"
    orphan.write_bytes(b"orphan")
    old = time.time() - (workspace.settings.orphan_grace_hours + 1) * 3600
    os.utime(orphan, (old, old))
    report = workspace.storage_report()
    assert any(item["ref"] == orphan.name for item in report["orphans"])
    assert workspace.cleanup_orphan_snapshots([orphan.name], dry_run=True) == [orphan.name]
    assert orphan.exists()
    assert workspace.cleanup_orphan_snapshots([orphan.name], dry_run=False) == [orphan.name]
    assert not orphan.exists()


def test_router_requires_tools_for_implementation(monkeypatch):
    clear_workflow_tables()
    seed_agent("analysis", "analysis-model", ["implementation"], [], 100)
    seed_agent("tools", "tools-model", ["implementation"], ["tools"], 10)
    monkeypatch.setattr(orchestrator, "discover_agents", lambda: [])
    assert orchestrator.choose_agent("implementation")["id"] == "tools"


def test_router_prefers_role_score_before_global_priority(monkeypatch):
    clear_workflow_tables()
    seed_agent("priority", "priority-model", ["research"], [], 100)
    seed_agent("specialist", "specialist-model", ["research"], [], 10)
    db.execute("update agent_profiles set role_scores_json=? where id='priority'", (json.dumps({"research": 50}),))
    db.execute("update agent_profiles set role_scores_json=? where id='specialist'", (json.dumps({"research": 90}),))
    monkeypatch.setattr(orchestrator, "discover_agents", lambda: [])
    assert orchestrator.choose_agent("research")["id"] == "specialist"


def test_run_preflight_rejects_missing_agents_before_staging(monkeypatch):
    clear_workflow_tables()
    target = WORKSPACES / "preflight-target"
    target.mkdir(exist_ok=True)
    (target / "app.py").write_text("print('safe')\n")
    monkeypatch.setattr(orchestrator, "provider_config", lambda _: ("key", "model"))
    monkeypatch.setattr(
        orchestrator,
        "choose_agent",
        lambda *_: (_ for _ in ()).throw(RuntimeError("No enabled research agent available")),
    )
    with pytest.raises(RuntimeError, match="No enabled research agent"):
        orchestrator.create_run("Preflight failure", "codex", target, False)
    assert not db.one("select id from runs where task='Preflight failure'")


def test_active_job_uniqueness():
    clear_workflow_tables()
    target = WORKSPACES / "unique-job"
    target.mkdir(exist_ok=True)
    run_id = "c" * 24
    now = db.utcnow()
    db.execute(
        "insert into runs(id,task,brain_provider,target_path,status,created_at,updated_at) values(?,?,?,?,?,?,?)",
        (run_id, "Unique job", "codex", str(target), "researching", now, now),
    )
    db.execute(
        "insert into jobs(run_id,job_type,status,created_at,updated_at) values(?,'research','pending',?,?)",
        (run_id, now, now),
    )
    with pytest.raises(sqlite3.IntegrityError):
        db.execute(
            "insert into jobs(run_id,job_type,status,created_at,updated_at) values(?,'research','running',?,?)",
            (run_id, now, now),
        )


def test_run_events_and_artifacts_are_bounded(monkeypatch):
    clear_workflow_tables()
    run_id = "e" * 24
    now = db.utcnow()
    db.execute(
        "insert into runs(id,task,brain_provider,target_path,status,created_at,updated_at) values(?,?,?,?,?,?,?)",
        (run_id, "Bound history", "codex", str(WORKSPACES), "researching", now, now),
    )
    monkeypatch.setattr(db, "MAX_RUN_EVENTS", 2)
    monkeypatch.setattr(orchestrator, "MAX_RUN_ARTIFACTS", 2)
    for index in range(3):
        db.add_event(run_id, "test", f"event {index}")
        orchestrator.save_artifact(run_id, "test", f"artifact {index}", str(index))
    assert [row["message"] for row in db.all_rows("select message from run_events order by id")] == ["event 1", "event 2"]
    assert [row["name"] for row in db.all_rows("select name from run_artifacts order by id")] == ["artifact 1", "artifact 2"]


def test_run_state_rejects_invalid_transition():
    validate_transition("researching", "awaiting_approval")
    with pytest.raises(RuntimeError, match="Invalid run transition"):
        validate_transition("completed", "implementing")


def test_shared_search_console_and_pinned_context():
    target = WORKSPACES / "shared-tools"
    target.mkdir(exist_ok=True)
    source = target / "source.py"
    source.write_text("needle one\nneedle two\n")
    result = search_text(target, "needle", limit=1)
    assert len(result["results"]) == 1
    assert result["next_cursor"] == 1
    assert run_check(target, "python", ["-c", "print('unsafe')"]) == "Rejected inline program"
    assert "PINNED FILE" in pinned_context(target, [str(source)])
    outside = WORKSPACES / "outside-context.txt"
    outside.write_text("outside")
    with pytest.raises(HTTPException) as exc:
        pinned_context(target, [str(outside)])
    assert exc.value.status_code == 400


def test_approval_creates_history_snapshot_and_approval_atomically(monkeypatch):
    clear_workflow_tables()
    seed_agent("impl", "implementation-model", ["implementation"], ["tools"], 100)
    target = WORKSPACES / "approval-target"
    target.mkdir(exist_ok=True)
    (target / "main.py").write_text("print('before')\n")
    run_id = "b" * 24
    stage_workspace(run_id, target)
    now = db.utcnow()
    db.execute(
        "insert into runs(id,task,brain_provider,brain_model,target_path,status,baseline_hash,created_at,updated_at) values(?,?,?,?,?,?,?,?,?)",
        (run_id, "Change output", "codex", "gpt-test", str(target), "awaiting_approval", manifest_hash(target), now, now),
    )
    monkeypatch.setattr(orchestrator, "start_job_queue", lambda: None)
    result = orchestrator.approve_run(run_id, "Approved plan")
    assert result["status"] == "implementing"
    assert db.one("select * from history_snippets where run_id=?", (run_id,))
    assert db.one("select * from run_approvals where run_id=?", (run_id,))
    assert db.one("select * from jobs where run_id=? and status='pending'", (run_id,))
    snapshot = db.one("select * from snapshots where id=?", (result["snapshot_id"],))
    assert snapshot and (TEST_ROOT / "snapshots" / snapshot["ref"]).is_file()
