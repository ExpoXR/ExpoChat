import json
import os
import secrets
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

from backend import brain_io, brain_runner, db, orchestrator, plan_graph, worker, workspace  # noqa: E402
from backend.main import pinned_context  # noqa: E402
from backend.migrations import MIGRATIONS  # noqa: E402
from backend.prompts import CAVEMAN_OUTPUT_INSTRUCTIONS  # noqa: E402
from backend.run_state import validate_transition  # noqa: E402
from backend.security import decrypt_secret, encrypt_secret  # noqa: E402
from backend.verification_policy import (  # noqa: E402
    PolicyDecision,
    evaluate_apply_gate,
    get_check_evidence,
    record_check_evidence,
)
from backend.workspace import (  # noqa: E402
    allowed_path,
    create_snapshot,
    manifest_hash,
    restore_snapshot,
    stage_workspace,
)
from backend.workspace_tools import TOOL_DEFS, execute_tool, run_check, search_text  # noqa: E402


def setup_module():
    db.init_db()


def clear_workflow_tables():
    for table in (
        "check_evidence", "brain_memory", "jobs", "subtask_results", "subtasks",
        "verification_results", "run_approvals", "history_snippets", "plan_versions",
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
    assert [row["version"] for row in versions] == [version for version, _ in MIGRATIONS]


def test_credential_round_trip():
    encrypted = encrypt_secret("secret-value")
    assert "secret-value" not in encrypted
    assert decrypt_secret(encrypted) == "secret-value"


def test_codex_runner_applies_api_key_before_starting_thread(monkeypatch):
    calls = []

    class FakeThread:
        def run(self, prompt):
            calls.append(("run", prompt))
            total = SimpleNamespace(
                input_tokens=10,
                cached_input_tokens=2,
                output_tokens=3,
                reasoning_output_tokens=1,
                total_tokens=13,
            )
            return SimpleNamespace(final_response="OK", usage=SimpleNamespace(total=total))

    class FakeCodex:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def login_api_key(self, api_key):
            calls.append(("login", api_key))

        def thread_start(self, *, model, sandbox, cwd, ephemeral, config):
            calls.append(("start", model, sandbox, cwd, ephemeral, config))
            return FakeThread()

    fake_module = SimpleNamespace(Codex=FakeCodex, Sandbox=SimpleNamespace(read_only="read-only"))
    monkeypatch.setitem(sys.modules, "openai_codex", fake_module)

    result = brain_runner.run_codex({"api_key": "secret", "model": "gpt-test", "prompt": "ping", "allow_web": True})

    assert result == {
        "content": "OK",
        "usage": {
            "input_tokens": 10,
            "cached_input_tokens": 2,
            "output_tokens": 3,
            "reasoning_output_tokens": 1,
            "total_tokens": 13,
        },
    }
    assert calls == [
        ("login", "secret"),
        ("start", "gpt-test", "read-only", os.environ.get("HOME"), True, {"web_search": "live"}),
        ("run", "ping"),
    ]


def test_gemini_runner_maps_usage_and_wires_web_search(monkeypatch):
    calls = {}

    class FakeModels:
        def generate_content(self, *, model, contents, config):
            calls.update(model=model, contents=contents, config=config)
            meta = SimpleNamespace(prompt_token_count=5, candidates_token_count=7, total_token_count=12)
            return SimpleNamespace(text="OK", usage_metadata=meta)

    class FakeClient:
        def __init__(self, api_key):
            calls["api_key"] = api_key
            self.models = FakeModels()

    fake_types = SimpleNamespace(
        GenerateContentConfig=lambda **kw: ("config", kw),
        Tool=lambda **kw: ("tool", kw),
        GoogleSearch=lambda: "search",
    )
    fake_genai = SimpleNamespace(Client=FakeClient, types=fake_types)
    monkeypatch.setitem(sys.modules, "google", SimpleNamespace(genai=fake_genai))
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", fake_types)

    result = brain_runner.run_gemini(
        {"api_key": "gk", "model": "gemini-x", "prompt": "ping", "allow_web": True}
    )

    assert result == {"content": "OK", "usage": {"input_tokens": 5, "output_tokens": 7, "total_tokens": 12}}
    assert calls["api_key"] == "gk"
    assert calls["model"] == "gemini-x"
    # allow_web wires a google_search tool into the generate config
    assert calls["config"][0] == "config"
    assert "tools" in calls["config"][1]


def test_gemini_provider_config_uses_stored_key():
    db.init_db()
    db.execute(
        "insert into brain_configs(provider,model,key_ciphertext,source,enabled,updated_at) "
        "values(?,?,?,?,?,?) on conflict(provider) do update set "
        "model=excluded.model,key_ciphertext=excluded.key_ciphertext,source=excluded.source,enabled=excluded.enabled",
        ("gemini", "gemini-2.5-pro", encrypt_secret("gk-123"), "stored", 1, db.utcnow()),
    )
    key, model = orchestrator.provider_config("gemini")
    assert key == "gk-123"
    assert model == "gemini-2.5-pro"


def test_migration_widens_brain_check_to_gemini():
    from backend.migrations import _brain_gemini

    conn = sqlite3.connect(":memory:")
    conn.executescript(
        "create table brain_configs ("
        " provider text primary key check(provider in ('codex','claude')),"
        " model text not null, key_ciphertext text, source text not null default 'environment',"
        " enabled integer not null default 0, validated_at text, last_error text, updated_at text not null);"
        "insert into brain_configs(provider,model,source,enabled,updated_at)"
        " values('codex','gpt-x','environment',1,'t0');"
    )
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("insert into brain_configs(provider,model,source,enabled,updated_at) values('gemini','g','e',1,'t')")

    _brain_gemini(conn)  # upgrade path
    _brain_gemini(conn)  # idempotent

    assert conn.execute("select model from brain_configs where provider='codex'").fetchone()[0] == "gpt-x"
    conn.execute("insert into brain_configs(provider,model,source,enabled,updated_at) values('gemini','g','environment',1,'t')")
    assert {r[0] for r in conn.execute("select provider from brain_configs")} == {"codex", "gemini"}


def test_stream_gemini_emits_incremental_tokens(monkeypatch, capsys):
    class FakeChunk:
        def __init__(self, text, meta=None):
            self.text = text
            self.usage_metadata = meta

    class FakeModels:
        def generate_content_stream(self, *, model, contents, config):
            yield FakeChunk("Hel")
            yield FakeChunk("lo", SimpleNamespace(prompt_token_count=2, candidates_token_count=1, total_token_count=3))

    class FakeClient:
        def __init__(self, api_key):
            self.models = FakeModels()

    fake_types = SimpleNamespace(GenerateContentConfig=lambda **kw: ("cfg", kw))
    fake_genai = SimpleNamespace(Client=FakeClient, types=fake_types)
    monkeypatch.setitem(sys.modules, "google", SimpleNamespace(genai=fake_genai))
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", fake_types)

    brain_runner.stream_gemini({"api_key": "k", "model": "m", "prompt": "hi"})
    lines = [json.loads(line) for line in capsys.readouterr().out.strip().splitlines()]
    assert lines[0] == {"token": "Hel"}
    assert lines[1] == {"token": "lo"}
    assert lines[-1]["done"] is True
    assert lines[-1]["usage"]["total_tokens"] == 3


def test_run_streaming_buffered_fallback_emits_token_then_done(monkeypatch, capsys):
    monkeypatch.setattr(brain_runner, "run_codex", lambda payload: {"content": "hello", "usage": {"total_tokens": 4}})
    brain_runner.run_streaming({"provider": "codex", "api_key": "k", "model": "m", "prompt": "p"})
    lines = [json.loads(line) for line in capsys.readouterr().out.strip().splitlines()]
    assert lines[0] == {"token": "hello"}
    assert lines[-1] == {"done": True, "usage": {"total_tokens": 4}}


def test_record_chat_usage_writes_paid_ledger():
    db.init_db()
    db.execute("delete from usage_ledger")
    orchestrator.record_chat_usage("chat", "gemini", {"input_tokens": 3, "output_tokens": 2})
    assert db.ledger_totals_today(paid_only=True)["total"] == 5


def test_settings_round_trip_and_defaults():
    db.init_db()
    db.execute("delete from app_settings where key='token_budget_daily'")
    db.execute("delete from app_settings where key='snapshot_retention_days'")
    assert db.get_setting_int("token_budget_daily") == 0  # falls back to default
    db.set_setting("token_budget_daily", 5000)
    db.set_setting("theme", "light")
    assert db.get_setting_int("token_budget_daily") == 5000
    settings_map = db.all_settings()
    assert settings_map["theme"] == "light"
    assert settings_map["token_budget_daily"] == "5000"
    assert settings_map["snapshot_retention_days"] == "30"


def test_record_usage_writes_paid_and_local_ledger_rows():
    db.init_db()
    db.execute("delete from usage_ledger")
    run_id = "run-ledger-" + secrets.token_hex(4)
    db.execute(
        "insert into runs(id,task,brain_provider,brain_model,target_path,status,created_at,updated_at) "
        "values(?,?,?,?,?,?,?,?)",
        (run_id, "t", "gemini", "gemini-2.5-pro", "/x", "researching", db.utcnow(), db.utcnow()),
    )
    orchestrator.record_usage(run_id, {"usage": {"input_tokens": 100, "output_tokens": 40, "total_tokens": 140}}, "brain")
    orchestrator.record_usage(run_id, {"usage": {"prompt_eval_count": 10, "eval_count": 5}}, "ollama")

    paid = db.ledger_totals_today(paid_only=True)
    everything = db.ledger_totals_today(paid_only=False)
    assert paid["total"] == 140  # ollama excluded from paid budget
    assert everything["total"] == 140 + 15
    providers = {row["provider"] for row in db.ledger_by_provider_today()}
    assert {"gemini", "ollama"} <= providers


def test_check_budget_blocks_over_daily_cap():
    db.init_db()
    db.execute("delete from usage_ledger")
    db.set_setting("token_budget_daily", "0")
    orchestrator.check_budget()  # unlimited: no raise
    db.record_ledger("brain", "gemini", 600, 400, 1000)
    db.set_setting("token_budget_daily", "800")
    with pytest.raises(orchestrator.BudgetExceeded):
        orchestrator.check_budget()
    db.set_setting("token_budget_daily", "5000")
    orchestrator.check_budget()  # under cap: no raise


def test_check_budget_blocks_over_run_cap():
    db.init_db()
    db.execute("delete from usage_ledger")
    db.set_setting("token_budget_daily", "0")
    db.set_setting("token_budget_run", "500")
    run_id = "run-cap-" + secrets.token_hex(4)
    db.execute(
        "insert into runs(id,task,brain_provider,brain_model,target_path,status,usage_json,created_at,updated_at) "
        "values(?,?,?,?,?,?,?,?,?)",
        (run_id, "t", "claude", "m", "/x", "researching", json.dumps({"brain": {"total_tokens": 700}}), db.utcnow(), db.utcnow()),
    )
    with pytest.raises(orchestrator.BudgetExceeded):
        orchestrator.check_budget(run_id)
    db.set_setting("token_budget_run", "0")
    orchestrator.check_budget(run_id)  # unlimited: no raise


def test_caveman_prompt_applies_to_brain_and_ollama(monkeypatch):
    captured = {}
    monkeypatch.setattr(orchestrator, "provider_config", lambda _: ("key", "model"))

    class FakeResponse:
        status_code = 200

        def json(self):
            return {"content": "OK", "usage": {}}

    class FakeClient:
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def post(self, _url, **kwargs):
            captured.update(kwargs["json"])
            return FakeResponse()

    monkeypatch.setattr(orchestrator.httpx, "Client", FakeClient)
    assert orchestrator.call_brain("codex", "Do work") == "OK"
    assert CAVEMAN_OUTPUT_INSTRUCTIONS in captured["prompt"]
    assert CAVEMAN_OUTPUT_INSTRUCTIONS in worker.agent_system_prompt("research")


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


def test_snapshot_restore_recovers_path_type_changes():
    clear_workflow_tables()
    file_target = WORKSPACES / "restore-file-type.txt"
    file_target.write_text("file snapshot")
    file_snapshot = create_snapshot(file_target)
    file_target.unlink()
    file_target.mkdir()
    (file_target / "wrong.txt").write_text("wrong type")
    restore_snapshot(file_snapshot["id"])
    assert file_target.is_file()
    assert file_target.read_text() == "file snapshot"

    directory_target = WORKSPACES / "restore-directory-type"
    directory_target.mkdir()
    (directory_target / "value.txt").write_text("directory snapshot")
    directory_snapshot = create_snapshot(directory_target)
    for child in directory_target.iterdir():
        child.unlink()
    directory_target.rmdir()
    directory_target.write_text("wrong type")
    restore_snapshot(directory_snapshot["id"])
    assert directory_target.is_dir()
    assert (directory_target / "value.txt").read_text() == "directory snapshot"


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


def test_run_events_and_artifacts_are_bounded():
    clear_workflow_tables()
    run_id = "e" * 24
    now = db.utcnow()
    db.execute(
        "insert into runs(id,task,brain_provider,target_path,status,created_at,updated_at) values(?,?,?,?,?,?,?)",
        (run_id, "Bound history", "codex", str(WORKSPACES), "researching", now, now),
    )
    try:
        db.set_setting("run_events_cap", 2)
        db.set_setting("run_artifacts_cap", 2)
        for index in range(3):
            db.add_event(run_id, "test", f"event {index}")
            orchestrator.save_artifact(run_id, "test", f"artifact {index}", str(index))
        assert [row["message"] for row in db.all_rows("select message from run_events order by id")] == ["event 1", "event 2"]
        assert [row["name"] for row in db.all_rows("select name from run_artifacts order by id")] == ["artifact 1", "artifact 2"]
    finally:
        db.set_setting("run_events_cap", 500)
        db.set_setting("run_artifacts_cap", 200)


def test_retention_cap_ignores_nonpositive_override():
    # A zero/invalid override must never wipe a table; it falls back to the default floor.
    try:
        db.set_setting("run_events_cap", 0)
        assert db.retention_cap("run_events_cap", db.MAX_RUN_EVENTS) == db.MAX_RUN_EVENTS
        db.set_setting("run_events_cap", 1200)
        assert db.retention_cap("run_events_cap", db.MAX_RUN_EVENTS) == 1200
    finally:
        db.set_setting("run_events_cap", 500)


def test_plan_versions_are_append_only_history():
    clear_workflow_tables()
    run_id = "p" * 24
    now = db.utcnow()
    db.execute(
        "insert into runs(id,task,brain_provider,target_path,status,created_at,updated_at) values(?,?,?,?,?,?,?)",
        (run_id, "Plan history", "codex", str(WORKSPACES), "awaiting_approval", now, now),
    )
    db.add_plan_version(run_id, "draft", "plan v1", "codex")
    db.add_plan_version(run_id, "edit", "plan v2 edited", "codex")
    db.add_plan_version(run_id, "approved", "plan v2 edited", "codex")
    versions = db.plan_versions(run_id)
    assert [row["version"] for row in versions] == [1, 2, 3]
    assert [row["kind"] for row in versions] == ["draft", "edit", "approved"]
    assert versions[0]["content"] == "plan v1"


def test_parse_task_graph_tolerates_fences_and_prose():
    fenced = "here is the graph\n```json\n{\"subtasks\": [{\"node_id\": \"a\", \"spec\": \"do a\"}]}\n```\nthanks"
    graph = plan_graph.parse_task_graph(fenced)
    assert graph["subtasks"][0]["node_id"] == "a"
    with pytest.raises(plan_graph.GraphError):
        plan_graph.parse_task_graph("no json here")


def test_validate_graph_orders_topologically_and_normalizes():
    graph = {
        "subtasks": [
            {"node_id": "b", "spec": "depends on a", "depends_on": ["a"], "file_globs": ["src/b/**"]},
            {"node_id": "a", "title": "First", "spec": "root", "file_globs": ["src/a/**"], "role": "implementation"},
        ]
    }
    nodes = plan_graph.validate_graph(graph)
    assert [n["node_id"] for n in nodes] == ["a", "b"]  # dependency before dependant
    assert nodes[0]["title"] == "First"


def test_validate_graph_rejects_cycles_and_bad_refs():
    with pytest.raises(plan_graph.GraphError):
        plan_graph.validate_graph({"subtasks": [
            {"node_id": "a", "spec": "x", "depends_on": ["b"]},
            {"node_id": "b", "spec": "y", "depends_on": ["a"]},
        ]})
    with pytest.raises(plan_graph.GraphError):
        plan_graph.validate_graph({"subtasks": [{"node_id": "a", "spec": "x", "depends_on": ["ghost"]}]})
    with pytest.raises(plan_graph.GraphError):
        plan_graph.validate_graph({"subtasks": []})


def test_independent_pairs_sharing_globs_flags_parallel_overlap():
    nodes = plan_graph.validate_graph({"subtasks": [
        {"node_id": "a", "spec": "x", "file_globs": ["src/shared.py"]},
        {"node_id": "b", "spec": "y", "file_globs": ["src/shared.py"]},
        {"node_id": "c", "spec": "z", "depends_on": ["a"], "file_globs": ["src/shared.py"]},
    ]})
    conflicts = plan_graph.independent_pairs_sharing_globs(nodes)
    # a and b are independent and share a glob → conflict; c depends on a so a-c is serialized.
    assert ("a", "b") in conflicts
    assert ("a", "c") not in conflicts


def test_insert_subtasks_persists_and_replaces_graph():
    clear_workflow_tables()
    run_id = "d" * 24
    now = db.utcnow()
    db.execute(
        "insert into runs(id,task,brain_provider,target_path,status,created_at,updated_at) values(?,?,?,?,?,?,?)",
        (run_id, "DAG run", "codex", str(WORKSPACES), "awaiting_approval", now, now),
    )
    nodes = plan_graph.validate_graph({"subtasks": [
        {"node_id": "a", "spec": "build a", "file_globs": ["a/**"]},
        {"node_id": "b", "spec": "build b", "depends_on": ["a"]},
    ]})
    db.insert_subtasks(run_id, nodes)
    persisted = db.subtasks(run_id)
    assert [row["node_id"] for row in persisted] == ["a", "b"]
    assert json.loads(persisted[1]["depends_on_json"]) == ["a"]

    db.update_subtask(persisted[0]["id"], status="done", result_summary="done a")
    db.add_subtask_result(persisted[0]["id"], run_id, "implementation", "transcript a")
    assert db.subtasks(run_id)[0]["status"] == "done"
    assert db.all_rows("select * from subtask_results where run_id=?", (run_id,))[0]["content"] == "transcript a"

    # Re-decomposition replaces the prior graph rather than duplicating node_ids.
    db.insert_subtasks(run_id, plan_graph.validate_graph({"subtasks": [{"node_id": "a", "spec": "only a"}]}))
    assert [row["node_id"] for row in db.subtasks(run_id)] == ["a"]


def test_two_subtask_jobs_can_be_active_but_types_stay_singleton():
    clear_workflow_tables()
    run_id = "g" * 24
    now = db.utcnow()
    db.execute(
        "insert into runs(id,task,brain_provider,target_path,status,created_at,updated_at) values(?,?,?,?,?,?,?)",
        (run_id, "Concurrent subtasks", "codex", str(WORKSPACES), "implementing", now, now),
    )
    # Two active subtask jobs (distinct node_id) are allowed simultaneously.
    db.execute(
        "insert into jobs(run_id,job_type,node_id,status,created_at,updated_at) values(?,?,?,?,?,?)",
        (run_id, "subtask", "a", "running", now, now),
    )
    db.execute(
        "insert into jobs(run_id,job_type,node_id,status,created_at,updated_at) values(?,?,?,?,?,?)",
        (run_id, "subtask", "b", "pending", now, now),
    )
    assert len(db.all_rows("select id from jobs where run_id=? and job_type='subtask'", (run_id,))) == 2
    # A second active job of the same non-subtask type is rejected by the partial unique index.
    db.execute(
        "insert into jobs(run_id,job_type,status,created_at,updated_at) values(?,?,?,?,?)",
        (run_id, "merge", "pending", now, now),
    )
    with pytest.raises(sqlite3.IntegrityError):
        db.execute(
            "insert into jobs(run_id,job_type,status,created_at,updated_at) values(?,?,?,?,?)",
            (run_id, "merge", "pending", now, now),
        )


def _make_base_stage(run_id: str, name: str) -> Path:
    target = WORKSPACES / name
    target.mkdir(exist_ok=True)
    (target / "base.txt").write_text("base\n")
    return stage_workspace(run_id, target)


def test_stage_subtask_isolates_worktrees():
    run_id = "h" * 24
    base = _make_base_stage(run_id, "dag-iso-target")
    wt_a = workspace.stage_subtask(run_id, "a", base)
    wt_b = workspace.stage_subtask(run_id, "b", base)
    (wt_a / "only_a.txt").write_text("a\n")
    assert (wt_a / "base.txt").read_text() == "base\n"
    assert not (wt_b / "only_a.txt").exists()  # b's worktree is unaffected
    assert not (base / "only_a.txt").exists()  # base stage is unaffected


def test_merge_worktrees_applies_disjoint_changes():
    run_id = "i" * 24
    base = _make_base_stage(run_id, "dag-merge-target")
    wt_a = workspace.stage_subtask(run_id, "a", base)
    wt_b = workspace.stage_subtask(run_id, "b", base)
    (wt_a / "a.txt").write_text("from a\n")
    (wt_b / "b.txt").write_text("from b\n")
    merged = workspace.merge_worktrees(base, [("a", wt_a), ("b", wt_b)])
    assert sorted(merged["changed"]) == ["a.txt", "b.txt"]
    assert (base / "a.txt").read_text() == "from a\n"
    assert (base / "b.txt").read_text() == "from b\n"


def test_merge_worktrees_raises_on_conflict_without_writing():
    run_id = "j" * 24
    base = _make_base_stage(run_id, "dag-conflict-target")
    wt_a = workspace.stage_subtask(run_id, "a", base)
    wt_b = workspace.stage_subtask(run_id, "b", base)
    (wt_a / "shared.txt").write_text("a wins\n")
    (wt_b / "shared.txt").write_text("b wins\n")
    with pytest.raises(workspace.MergeConflict) as info:
        workspace.merge_worktrees(base, [("a", wt_a), ("b", wt_b)])
    assert info.value.conflicts[0]["path"] == "shared.txt"
    assert not (base / "shared.txt").exists()  # nothing written on conflict


def test_execute_dag_runs_subtasks_and_merges(monkeypatch):
    clear_workflow_tables()
    run_id = "k" * 24
    now = db.utcnow()
    db.execute(
        "insert into runs(id,task,brain_provider,target_path,status,created_at,updated_at) values(?,?,?,?,?,?,?)",
        (run_id, "DAG execute", "codex", str(WORKSPACES), "implementing", now, now),
    )
    base = _make_base_stage(run_id, "dag-exec-target")
    nodes = plan_graph.validate_graph({"subtasks": [
        {"node_id": "a", "spec": "make a", "file_globs": ["a.txt"]},
        {"node_id": "b", "spec": "make b", "depends_on": ["a"], "file_globs": ["b.txt"]},
    ]})
    db.insert_subtasks(run_id, nodes)

    def fake_worker_call(rid, model, mode, task, workspace="workspace", max_turns=24, node_id=None):
        node = Path(workspace).parts[1] if workspace.startswith("subtasks/") else "main"
        (orchestrator.settings.jobs_dir / rid / workspace / f"{node}.txt").write_text(f"from {node}\n")
        return {"ok": True, "content": f"did {node}", "usage": {}}

    monkeypatch.setattr(orchestrator, "worker_call", fake_worker_call)
    monkeypatch.setattr(orchestrator, "choose_agent", lambda *a, **k: {"id": "impl", "model": "m", "name": "Impl"})
    monkeypatch.setattr(orchestrator, "record_usage", lambda *a, **k: None)

    summary, implementer_ids = orchestrator.execute_dag(run_id, base)
    assert (base / "a.txt").read_text() == "from a\n"
    assert (base / "b.txt").read_text() == "from b\n"
    assert implementer_ids == {"impl"}
    assert "did a" in summary and "did b" in summary
    assert all(row["status"] == "done" for row in db.subtasks(run_id))


def test_worker_activity_records_safe_live_file_states():
    clear_workflow_tables()
    run_id = "f" * 24
    now = db.utcnow()
    db.execute(
        "insert into runs(id,task,brain_provider,target_path,status,created_at,updated_at) values(?,?,?,?,?,?,?)",
        (run_id, "Live activity", "codex", str(WORKSPACES), "implementing", now, now),
    )
    orchestrator.record_worker_activity(
        run_id,
        "implementation",
        {"type": "tool.started", "turn": 2, "name": "replace_text", "args": {"path": "src/app.py", "old": "secret"}},
    )
    orchestrator.record_worker_activity(
        run_id,
        "implementation",
        {"type": "tool.completed", "turn": 2, "name": "replace_text", "args": {"path": "src/app.py"}, "result": "Updated src/app.py"},
    )
    rows = db.all_rows("select event_type,message,data_json from run_events where run_id=? order by id", (run_id,))
    assert [json.loads(row["data_json"])["state"] for row in rows] == ["working", "changed"]
    assert all("secret" not in row["data_json"] for row in rows)


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
    assert run_check(target, "grep", ["needle", "../outside-context.txt"]) == "Rejected path traversal"
    assert run_check(target, "grep", ["needle", "--include=/etc/passwd"]) == "Rejected path outside staged workspace"
    assert "PINNED FILE" in pinned_context(target, [str(source)])
    outside = WORKSPACES / "outside-context.txt"
    outside.write_text("outside")
    with pytest.raises(HTTPException) as exc:
        pinned_context(target, [str(outside)])
    assert exc.value.status_code == 400


def test_all_worker_tools_are_registered_and_enforce_write_mode():
    target = WORKSPACES / "worker-tools"
    target.mkdir(exist_ok=True)
    names = {tool["function"]["name"] for tool in TOOL_DEFS}
    assert names == {"list_files", "read_file", "search_files", "write_file", "replace_text", "delete_file", "run_check"}
    assert execute_tool(target, "write_file", {"path": "notes.txt", "content": "alpha\nbeta\n"}, False) == "Rejected: worker is read-only"
    assert execute_tool(target, "write_file", {"path": "notes.txt", "content": "alpha\nbeta\n"}, True) == "Wrote notes.txt"
    assert "notes.txt" in execute_tool(target, "list_files", {}, False)
    assert "1: alpha" in execute_tool(target, "read_file", {"path": "notes.txt"}, False)
    assert "notes.txt:2:beta" in execute_tool(target, "search_files", {"pattern": "beta"}, False)
    assert execute_tool(
        target,
        "replace_text",
        {"path": "notes.txt", "old": "beta", "new": "gamma"},
        True,
    ) == "Updated notes.txt (1 replacement(s))"
    assert execute_tool(target, "delete_file", {"path": "notes.txt"}, True) == "Deleted notes.txt"
    assert not (target / "notes.txt").exists()


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


def test_plan_can_be_edited_and_redone(monkeypatch):
    clear_workflow_tables()
    seed_agent("research", "research-model", ["research"], [], 100)
    target = WORKSPACES / "redo-target"
    target.mkdir(exist_ok=True)
    (target / "main.py").write_text("print('redo')\n")
    run_id = "f" * 24
    now = db.utcnow()
    db.execute(
        "insert into runs(id,task,brain_provider,target_path,status,baseline_hash,draft_plan,created_at,updated_at) values(?,?,?,?,?,?,?,?,?)",
        (run_id, "Redo plan", "codex", str(target), "awaiting_approval", manifest_hash(target), "Old plan", now, now),
    )
    monkeypatch.setattr(orchestrator, "start_job_queue", lambda: None)

    edited = orchestrator.edit_plan(run_id, "Changed plan")
    assert edited["draft_plan"] == "Changed plan"
    assert db.one("select id from run_artifacts where run_id=? and kind='plan_edit'", (run_id,))

    redone = orchestrator.redo_plan(run_id)
    assert redone["status"] == "researching"
    assert redone["draft_plan"] is None
    assert db.one("select id from jobs where run_id=? and job_type='research' and status='pending'", (run_id,))


def test_resume_researches_again_when_workspace_changed(monkeypatch):
    clear_workflow_tables()
    target = WORKSPACES / "resume-stale-target"
    target.mkdir(exist_ok=True)
    source = target / "main.py"
    source.write_text("print('before')\n")
    baseline = manifest_hash(target)
    snapshot = create_snapshot(target)
    run_id = "9" * 24
    stage_workspace(run_id, target)
    source.write_text("print('external change')\n")
    now = db.utcnow()
    db.execute(
        "insert into runs(id,task,brain_provider,target_path,status,baseline_hash,approved_plan,snapshot_id,created_at,updated_at) values(?,?,?,?,?,?,?,?,?,?)",
        (run_id, "Resume stale run", "codex", str(target), "failed", baseline, "Old plan", snapshot["id"], now, now),
    )
    monkeypatch.setattr(orchestrator, "start_job_queue", lambda: None)

    orchestrator.resume_run(run_id)

    resumed = db.one("select * from runs where id=?", (run_id,))
    assert resumed["status"] == "researching"
    assert resumed["approved_plan"] is None
    assert resumed["snapshot_id"] is None
    assert resumed["baseline_hash"] == manifest_hash(target)
    assert (orchestrator.settings.jobs_dir / run_id / "workspace" / "main.py").read_text() == "print('external change')\n"
    assert db.one("select id from jobs where run_id=? and job_type='research' and status='pending'", (run_id,))


# ---------------------------------------------------------------------------
# Phase 3: brain_io structured I/O
# ---------------------------------------------------------------------------

def test_extract_json_handles_fences_and_prose():
    # Fenced JSON
    fenced = '```json\n{"passed": true, "verdict": "ok"}\n```'
    result = brain_io.extract_json(fenced)
    assert result["passed"] is True

    # Prose around JSON
    prose = 'Here is my analysis:\n\n{"key": "value"}\n\nHope that helps!'
    result = brain_io.extract_json(prose)
    assert result["key"] == "value"

    # Clean JSON
    clean = '{"subtasks": [{"node_id": "a"}]}'
    result = brain_io.extract_json(clean)
    assert result["subtasks"][0]["node_id"] == "a"

    # No JSON
    with pytest.raises(ValueError, match="No JSON"):
        brain_io.extract_json("no json here at all")

    # Empty
    with pytest.raises(ValueError, match="Empty"):
        brain_io.extract_json("")


def test_parse_verdict_typed_result():
    # Well-formed JSON verdict
    raw = '{"passed": true, "verdict": "all good", "repair_task": "", "scope_expansion": false}'
    v = brain_io.parse_verdict(raw)
    assert v.passed is True
    assert v.verdict == "all good"
    assert v.scope_expansion is False
    assert '"passed": true' in v.to_json()

    # Fenced JSON
    fenced = '```json\n{"passed": false, "verdict": "bugs", "repair_task": "fix it", "scope_expansion": true}\n```'
    v = brain_io.parse_verdict(fenced)
    assert v.passed is False
    assert v.repair_task == "fix it"
    assert v.scope_expansion is True

    # Fallback: no JSON, string-sniff
    raw_pass = 'Here is the verdict: "passed":true somewhere'
    v = brain_io.parse_verdict(raw_pass)
    assert v.passed is True
    assert v.verdict == raw_pass  # raw text preserved

    raw_fail = "The implementation has bugs."
    v = brain_io.parse_verdict(raw_fail)
    assert v.passed is False


def test_build_prompts_contain_expected_structure():
    plan = brain_io.build_plan_prompt("build API", "dossier content")
    assert "USER TASK:" in plan and "build API" in plan
    assert "WORKER DOSSIER:" in plan and "dossier content" in plan

    decompose = brain_io.build_decompose_prompt("the plan")
    assert "suggested_model" in decompose  # new field in schema
    assert "the plan" in decompose

    verdict = brain_io.build_verdict_prompt("plan text", ["report1", "report2"])
    assert "PLAN:" in verdict and "REPORTS:" in verdict

    verify = brain_io.build_verification_prompt("plan text", "summary text")
    assert "PASS or FAIL" in verify


# ---------------------------------------------------------------------------
# Phase 3: brain memory
# ---------------------------------------------------------------------------

def test_brain_memory_accumulates_and_truncates():
    clear_workflow_tables()
    run_id = "m" * 24
    now = db.utcnow()
    db.execute(
        "insert into runs(id,task,brain_provider,target_path,status,created_at,updated_at) values(?,?,?,?,?,?,?)",
        (run_id, "Memory test", "codex", str(WORKSPACES), "researching", now, now),
    )

    # Accumulate entries
    db.add_brain_memory(run_id, "plan", "prompt", "What should we build?")
    db.add_brain_memory(run_id, "plan", "response", "Build a REST API with auth.")
    db.add_brain_memory(run_id, "decompose", "prompt", "Break into subtasks.")
    db.add_brain_memory(run_id, "decompose", "response", "Two subtasks: auth + endpoints.")

    entries = db.brain_memory(run_id)
    assert len(entries) == 4
    assert entries[0]["seq"] == 1
    assert entries[3]["seq"] == 4
    assert entries[0]["step"] == "plan"
    assert entries[2]["role"] == "prompt"
    assert all(e["tokens_estimate"] > 0 for e in entries)

    # Digest should contain prior context
    digest = orchestrator._build_memory_digest(run_id)
    assert "PRIOR CONTEXT" in digest
    assert "plan/prompt" in digest
    assert "Build a REST API" in digest

    # Empty run has no digest
    empty_id = "n" * 24
    db.execute(
        "insert into runs(id,task,brain_provider,target_path,status,created_at,updated_at) values(?,?,?,?,?,?,?)",
        (empty_id, "No memory", "codex", str(WORKSPACES), "researching", now, now),
    )
    assert orchestrator._build_memory_digest(empty_id) == ""


def test_brain_memory_budget_truncation():
    clear_workflow_tables()
    run_id = "t" * 24
    now = db.utcnow()
    db.execute(
        "insert into runs(id,task,brain_provider,target_path,status,created_at,updated_at) values(?,?,?,?,?,?,?)",
        (run_id, "Budget test", "codex", str(WORKSPACES), "researching", now, now),
    )
    # Set very tight budget
    db.set_setting("brain_memory_budget", "10")  # 10 tokens = 40 chars

    # Add entries that exceed budget
    db.add_brain_memory(run_id, "plan", "prompt", "A" * 100)
    db.add_brain_memory(run_id, "plan", "response", "B" * 100)
    db.add_brain_memory(run_id, "decompose", "prompt", "C" * 30)

    digest = orchestrator._build_memory_digest(run_id)
    # Should have dropped oldest entries to stay within budget
    assert "PRIOR CONTEXT" in digest
    # Last entry should always be present
    assert "C" * 30 in digest

    # Reset
    db.set_setting("brain_memory_budget", "4000")


# ---------------------------------------------------------------------------
# Phase 3: per-subtask agent selection
# ---------------------------------------------------------------------------

def test_choose_subtask_agent_honors_role(monkeypatch):
    clear_workflow_tables()
    seed_agent("researcher", "research-model", ["research"], [], 80)
    seed_agent("implementer", "impl-model", ["implementation"], ["tools"], 80)
    monkeypatch.setattr(orchestrator, "discover_agents", lambda: [])

    # Research-role subtask should pick the research agent
    node = {"role": "research", "node_id": "a"}
    agent = orchestrator.choose_subtask_agent(node)
    assert agent["id"] == "researcher"

    # Implementation-role subtask should pick the implementation agent
    node = {"role": "implementation", "node_id": "b"}
    agent = orchestrator.choose_subtask_agent(node)
    assert agent["id"] == "implementer"


def test_choose_subtask_agent_suggested_model(monkeypatch):
    clear_workflow_tables()
    seed_agent("default", "default-model", ["implementation"], ["tools"], 90)
    seed_agent("special", "special-model", ["implementation"], ["tools"], 50)
    monkeypatch.setattr(orchestrator, "discover_agents", lambda: [])

    # Without hint, picks highest priority
    node = {"role": "implementation", "node_id": "a"}
    agent = orchestrator.choose_subtask_agent(node)
    assert agent["id"] == "default"

    # With suggested_model, picks that specific model
    node = {"role": "implementation", "node_id": "b", "suggested_model": "special-model"}
    agent = orchestrator.choose_subtask_agent(node)
    assert agent["id"] == "special"

    # Unknown suggested_model falls back to default selection
    node = {"role": "implementation", "node_id": "c", "suggested_model": "nonexistent-model"}
    agent = orchestrator.choose_subtask_agent(node)
    assert agent["id"] == "default"


def test_round_robin_distributes_across_equal_agents(monkeypatch):
    clear_workflow_tables()
    orchestrator._subtask_assignment_counts.clear()
    seed_agent("agent-a", "model-a", ["implementation"], ["tools"], 80)
    seed_agent("agent-b", "model-b", ["implementation"], ["tools"], 80)
    # Equal role scores
    db.execute("update agent_profiles set role_scores_json=? where id='agent-a'", (json.dumps({"implementation": 80}),))
    db.execute("update agent_profiles set role_scores_json=? where id='agent-b'", (json.dumps({"implementation": 80}),))
    monkeypatch.setattr(orchestrator, "discover_agents", lambda: [])

    run_id = "r" * 24
    assignments = []
    for i in range(4):
        node = {"role": "implementation", "node_id": f"task-{i}"}
        agent = orchestrator.choose_subtask_agent(node, run_id=run_id)
        assignments.append(agent["id"])

    # Both agents should be used (round-robin distributes)
    assert "agent-a" in assignments
    assert "agent-b" in assignments
    # Each should get roughly equal assignments
    assert abs(assignments.count("agent-a") - assignments.count("agent-b")) <= 1
    orchestrator._subtask_assignment_counts.clear()


def test_validate_graph_passes_through_suggested_model():
    graph = {
        "subtasks": [
            {"node_id": "a", "spec": "do a", "role": "implementation", "suggested_model": "qwen3-coder"},
            {"node_id": "b", "spec": "do b", "role": "research", "suggested_model": None},
            {"node_id": "c", "spec": "do c"},
        ]
    }
    nodes = plan_graph.validate_graph(graph)
    assert nodes[0]["suggested_model"] == "qwen3-coder"
    assert nodes[1]["suggested_model"] is None
    assert nodes[2]["suggested_model"] is None


def test_execute_dag_uses_subtask_role(monkeypatch):
    """Integration: DAG execution uses per-subtask agent selection with mixed roles."""
    clear_workflow_tables()
    orchestrator._subtask_assignment_counts.clear()
    run_id = "d" * 24
    now = db.utcnow()
    db.execute(
        "insert into runs(id,task,brain_provider,target_path,status,created_at,updated_at) values(?,?,?,?,?,?,?)",
        (run_id, "Mixed role DAG", "codex", str(WORKSPACES), "implementing", now, now),
    )
    base = _make_base_stage(run_id, "dag-role-target")
    nodes = plan_graph.validate_graph({"subtasks": [
        {"node_id": "research-step", "spec": "research first", "role": "research", "file_globs": ["r.txt"]},
        {"node_id": "impl-step", "spec": "implement it", "depends_on": ["research-step"], "file_globs": ["i.txt"]},
    ]})
    db.insert_subtasks(run_id, nodes)
    seed_agent("res-agent", "res-model", ["research"], [], 80)
    seed_agent("impl-agent", "impl-model", ["implementation"], ["tools"], 80)

    chosen_agents = []

    def fake_choose_subtask(node, exclude=None, run_id=None):
        role = node.get("role", "implementation")
        if role == "research":
            agent = {"id": "res-agent", "model": "res-model", "name": "ResAgent"}
        else:
            agent = {"id": "impl-agent", "model": "impl-model", "name": "ImplAgent"}
        chosen_agents.append((node["node_id"], agent["id"]))
        return agent

    def fake_worker_call(rid, model, mode, task, workspace="workspace", max_turns=24, node_id=None):
        node = Path(workspace).parts[1] if workspace.startswith("subtasks/") else "main"
        (orchestrator.settings.jobs_dir / rid / workspace / f"{node}.txt").write_text(f"from {node}\n")
        return {"ok": True, "content": f"did {node}", "usage": {}}

    monkeypatch.setattr(orchestrator, "choose_subtask_agent", fake_choose_subtask)
    monkeypatch.setattr(orchestrator, "worker_call", fake_worker_call)
    monkeypatch.setattr(orchestrator, "record_usage", lambda *a, **k: None)

    summary, implementer_ids = orchestrator.execute_dag(run_id, base)
    # Research subtask should have been assigned res-agent
    assert ("research-step", "res-agent") in chosen_agents
    # Implementation subtask should have been assigned impl-agent
    assert ("impl-step", "impl-agent") in chosen_agents
    orchestrator._subtask_assignment_counts.clear()


def test_migration_10_creates_brain_memory_table():
    """Migration 10 creates brain_memory table and subtasks.suggested_model column."""
    with db.connect() as conn:
        tables = {row[0] for row in conn.execute(
            "select name from sqlite_master where type='table'"
        ).fetchall()}
        assert "brain_memory" in tables
        cols = {row[1] for row in conn.execute("pragma table_info(subtasks)")}
        assert "suggested_model" in cols


def test_migration_11_adds_acceptance_criteria():
    """Migration 11 adds acceptance_criteria column to subtasks."""
    with db.connect() as conn:
        cols = {row[1] for row in conn.execute("pragma table_info(subtasks)")}
        assert "acceptance_criteria" in cols


def test_insert_subtasks_stores_acceptance_criteria():
    clear_workflow_tables()
    seed_agent("agent-ac", "model-ac", ["implementation"], ["tools"])
    now = db.utcnow()
    db.execute(
        "insert into runs(id,task,brain_provider,target_path,status,created_at,updated_at) "
        "values(?,?,?,?,?,?,?)", ("run-ac", "t", "codex", "/tmp", "implementing", now, now),
    )
    nodes = [
        {"node_id": "n1", "title": "Step 1", "spec": "do X", "depends_on": [],
         "file_globs": [], "acceptance_criteria": "X works", "role": "implementation",
         "suggested_model": None},
    ]
    db.insert_subtasks("run-ac", nodes)
    stored = db.subtasks("run-ac")
    assert len(stored) == 1
    assert stored[0]["acceptance_criteria"] == "X works"


def test_enqueue_job_with_node_id(monkeypatch):
    clear_workflow_tables()
    monkeypatch.setattr(orchestrator, "start_job_queue", lambda: None)
    now = db.utcnow()
    db.execute(
        "insert into runs(id,task,brain_provider,target_path,status,created_at,updated_at) "
        "values(?,?,?,?,?,?,?)", ("run-nid", "t", "codex", "/tmp", "implementing", now, now),
    )
    orchestrator.enqueue_job("run-nid", "subtask", node_id="step-a")
    jobs = db.all_rows("select * from jobs where run_id='run-nid'")
    assert len(jobs) == 1
    assert jobs[0]["job_type"] == "subtask"
    assert jobs[0]["node_id"] == "step-a"
    # Duplicate enqueue same node_id should not create second job
    orchestrator.enqueue_job("run-nid", "subtask", node_id="step-a")
    jobs = db.all_rows("select * from jobs where run_id='run-nid'")
    assert len(jobs) == 1


def test_enqueue_ready_subtasks_chains_dag(monkeypatch):
    """_enqueue_ready_subtasks enqueues ready nodes and merge when all done."""
    clear_workflow_tables()
    monkeypatch.setattr(orchestrator, "start_job_queue", lambda: None)
    now = db.utcnow()
    db.execute(
        "insert into runs(id,task,brain_provider,target_path,status,created_at,updated_at) "
        "values(?,?,?,?,?,?,?)", ("run-rdy", "t", "codex", "/tmp", "implementing", now, now),
    )
    nodes = [
        {"node_id": "a", "title": "A", "spec": "do A", "depends_on": [],
         "file_globs": [], "acceptance_criteria": "", "role": "implementation", "suggested_model": None},
        {"node_id": "b", "title": "B", "spec": "do B", "depends_on": ["a"],
         "file_globs": [], "acceptance_criteria": "", "role": "implementation", "suggested_model": None},
    ]
    db.insert_subtasks("run-rdy", nodes)
    # Initially only "a" should be enqueued (no deps)
    orchestrator._enqueue_ready_subtasks("run-rdy")
    jobs = db.all_rows("select * from jobs where run_id='run-rdy' order by id")
    assert len(jobs) == 1
    assert jobs[0]["node_id"] == "a"
    assert jobs[0]["job_type"] == "subtask"
    # Mark "a" done, enqueue again — "b" should appear
    db.update_subtask("sub-run-rdy-a", status="done")
    orchestrator._enqueue_ready_subtasks("run-rdy")
    jobs = db.all_rows("select * from jobs where run_id='run-rdy' and status='pending' order by id")
    assert any(j["node_id"] == "b" for j in jobs)
    # Mark "b" done — merge job should appear
    db.update_subtask("sub-run-rdy-b", status="done")
    orchestrator._enqueue_ready_subtasks("run-rdy")
    merge_jobs = db.all_rows("select * from jobs where run_id='run-rdy' and job_type='merge'")
    assert len(merge_jobs) == 1


def test_subtask_recovery_on_init():
    """init_db resets running subtasks to pending."""
    clear_workflow_tables()
    now = db.utcnow()
    db.execute(
        "insert into runs(id,task,brain_provider,target_path,status,created_at,updated_at) "
        "values(?,?,?,?,?,?,?)", ("run-rec", "t", "codex", "/tmp", "implementing", now, now),
    )
    nodes = [
        {"node_id": "x", "title": "X", "spec": "do X", "depends_on": [],
         "file_globs": [], "acceptance_criteria": "", "role": "implementation", "suggested_model": None},
    ]
    db.insert_subtasks("run-rec", nodes)
    db.update_subtask("sub-run-rec-x", status="running")
    assert db.subtasks("run-rec")[0]["status"] == "running"
    db.init_db()
    assert db.subtasks("run-rec")[0]["status"] == "pending"


def test_implement_run_dispatches_subtask_jobs(monkeypatch):
    """Multi-agent implement_run enqueues subtask jobs and returns without blocking."""
    clear_workflow_tables()
    seed_agent("impl-dur", "dur-model", ["implementation"], ["tools"])
    now = db.utcnow()
    db.execute(
        "insert into runs(id,task,brain_provider,target_path,status,approved_plan,baseline_hash,"
        "implementation_agent_id,created_at,updated_at) "
        "values(?,?,?,?,?,?,?,?,?,?)",
        ("run-dur", "t", "codex", "/tmp", "implementing", "plan", "abc", "impl-dur", now, now),
    )
    nodes = [
        {"node_id": "s1", "title": "S1", "spec": "spec1", "depends_on": [],
         "file_globs": [], "acceptance_criteria": "", "role": "implementation", "suggested_model": None},
        {"node_id": "s2", "title": "S2", "spec": "spec2", "depends_on": ["s1"],
         "file_globs": [], "acceptance_criteria": "", "role": "implementation", "suggested_model": None},
    ]
    db.insert_subtasks("run-dur", nodes)
    # Prevent start_job_queue from actually draining
    monkeypatch.setattr(orchestrator, "start_job_queue", lambda: None)
    orchestrator.implement_run("run-dur")
    # Should have enqueued subtask job for s1 (ready), not s2 (blocked by s1)
    jobs = db.all_rows("select * from jobs where run_id='run-dur' and job_type='subtask'")
    assert len(jobs) == 1
    assert jobs[0]["node_id"] == "s1"
    # Run status should still be implementing (not verifying — didn't block)
    run = db.one("select status from runs where id='run-dur'")
    assert run["status"] == "implementing"


# -- Phase 5: Subtask resilience tests --

def test_transitive_deps():
    """_transitive_deps returns all transitive dependencies."""
    dep_map = {"a": [], "b": ["a"], "c": ["b"], "d": ["a"]}
    assert orchestrator._transitive_deps("c", dep_map) == {"a", "b"}
    assert orchestrator._transitive_deps("d", dep_map) == {"a"}
    assert orchestrator._transitive_deps("a", dep_map) == set()


def test_enqueue_ready_subtasks_partial_dag(monkeypatch):
    """Independent branches continue past a failed sibling."""
    clear_workflow_tables()
    seed_agent("a1", "m1", ["implementation"], ["tools"])
    now = db.utcnow()
    db.execute(
        "insert into runs(id,task,brain_provider,target_path,status,approved_plan,baseline_hash,"
        "implementation_agent_id,created_at,updated_at) values(?,?,?,?,?,?,?,?,?,?)",
        ("run-partial", "t", "codex", "/tmp", "implementing", "plan", "abc", "a1", now, now),
    )
    nodes = [
        {"node_id": "root", "title": "Root", "spec": "s", "depends_on": [],
         "file_globs": [], "acceptance_criteria": "", "role": "implementation", "suggested_model": None},
        {"node_id": "branch-a", "title": "Branch A", "spec": "s", "depends_on": ["root"],
         "file_globs": [], "acceptance_criteria": "", "role": "implementation", "suggested_model": None},
        {"node_id": "branch-b", "title": "Branch B", "spec": "s", "depends_on": ["root"],
         "file_globs": [], "acceptance_criteria": "", "role": "implementation", "suggested_model": None},
    ]
    db.insert_subtasks("run-partial", nodes)
    subs = db.subtasks("run-partial")
    db.update_subtask(next(s["id"] for s in subs if s["node_id"] == "root"), status="done")
    db.update_subtask(next(s["id"] for s in subs if s["node_id"] == "branch-a"), status="failed")
    monkeypatch.setattr(orchestrator, "start_job_queue", lambda: None)
    orchestrator._enqueue_ready_subtasks("run-partial")
    jobs = db.all_rows("select * from jobs where run_id='run-partial' and job_type='subtask'")
    assert len(jobs) == 1
    assert jobs[0]["node_id"] == "branch-b"


def test_enqueue_ready_subtasks_partial_merge(monkeypatch):
    """All non-failed branches done + failed branch → triggers merge."""
    clear_workflow_tables()
    seed_agent("a2", "m2", ["implementation"], ["tools"])
    now = db.utcnow()
    db.execute(
        "insert into runs(id,task,brain_provider,target_path,status,approved_plan,baseline_hash,"
        "implementation_agent_id,created_at,updated_at) values(?,?,?,?,?,?,?,?,?,?)",
        ("run-pm", "t", "codex", "/tmp", "implementing", "plan", "abc", "a2", now, now),
    )
    nodes = [
        {"node_id": "ok-node", "title": "OK", "spec": "s", "depends_on": [],
         "file_globs": [], "acceptance_criteria": "", "role": "implementation", "suggested_model": None},
        {"node_id": "bad-node", "title": "Bad", "spec": "s", "depends_on": [],
         "file_globs": [], "acceptance_criteria": "", "role": "implementation", "suggested_model": None},
    ]
    db.insert_subtasks("run-pm", nodes)
    subs = db.subtasks("run-pm")
    db.update_subtask(next(s["id"] for s in subs if s["node_id"] == "ok-node"), status="done")
    db.update_subtask(next(s["id"] for s in subs if s["node_id"] == "bad-node"), status="failed")
    monkeypatch.setattr(orchestrator, "start_job_queue", lambda: None)
    orchestrator._enqueue_ready_subtasks("run-pm")
    merge_jobs = db.all_rows("select * from jobs where run_id='run-pm' and job_type='merge'")
    assert len(merge_jobs) == 1


def test_subtask_retry_on_failure(monkeypatch):
    """Failed subtask with remaining attempts re-enqueues instead of failing run."""
    clear_workflow_tables()
    seed_agent("retry-agent", "retry-model", ["implementation"], ["tools"])
    now = db.utcnow()
    db.execute(
        "insert into runs(id,task,brain_provider,target_path,status,approved_plan,baseline_hash,"
        "implementation_agent_id,created_at,updated_at) values(?,?,?,?,?,?,?,?,?,?)",
        ("run-retry", "t", "codex", "/tmp", "implementing", "plan", "abc", "retry-agent", now, now),
    )
    nodes = [
        {"node_id": "retry-node", "title": "Retry Me", "spec": "s", "depends_on": [],
         "file_globs": [], "acceptance_criteria": "", "role": "implementation", "suggested_model": None},
    ]
    db.insert_subtasks("run-retry", nodes)
    monkeypatch.setattr(orchestrator, "start_job_queue", lambda: None)
    monkeypatch.setattr(orchestrator, "_run_subtask", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("test fail")))
    job = {"run_id": "run-retry", "node_id": "retry-node"}
    orchestrator._run_durable_subtask("run-retry", job)
    run = db.one("select status from runs where id='run-retry'")
    assert run["status"] == "implementing"
    sub = next(s for s in db.subtasks("run-retry") if s["node_id"] == "retry-node")
    assert sub["status"] == "pending"
    retry_jobs = db.all_rows("select * from jobs where run_id='run-retry' and job_type='subtask' and status='pending'")
    assert len(retry_jobs) == 1
    events = db.all_rows("select * from run_events where run_id='run-retry' and event_type='subtask.retry'")
    assert len(events) == 1


def test_subtask_fails_after_max_attempts(monkeypatch):
    """Subtask that exhausted retries is marked failed, run stays implementing if others pending."""
    clear_workflow_tables()
    db.set_setting("subtask_max_attempts", 2)
    seed_agent("exhaust-agent", "m", ["implementation"], ["tools"])
    now = db.utcnow()
    db.execute(
        "insert into runs(id,task,brain_provider,target_path,status,approved_plan,baseline_hash,"
        "implementation_agent_id,created_at,updated_at) values(?,?,?,?,?,?,?,?,?,?)",
        ("run-exhaust", "t", "codex", "/tmp", "implementing", "plan", "abc", "exhaust-agent", now, now),
    )
    nodes = [
        {"node_id": "exhaust-node", "title": "Exhaust", "spec": "s", "depends_on": [],
         "file_globs": [], "acceptance_criteria": "", "role": "implementation", "suggested_model": None},
        {"node_id": "other-node", "title": "Other", "spec": "s", "depends_on": [],
         "file_globs": [], "acceptance_criteria": "", "role": "implementation", "suggested_model": None},
    ]
    db.insert_subtasks("run-exhaust", nodes)
    sub_exhaust = next(s for s in db.subtasks("run-exhaust") if s["node_id"] == "exhaust-node")
    db.update_subtask(sub_exhaust["id"], attempts=1)
    monkeypatch.setattr(orchestrator, "start_job_queue", lambda: None)
    monkeypatch.setattr(orchestrator, "_run_subtask", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("final fail")))
    job = {"run_id": "run-exhaust", "node_id": "exhaust-node"}
    orchestrator._run_durable_subtask("run-exhaust", job)
    sub = next(s for s in db.subtasks("run-exhaust") if s["node_id"] == "exhaust-node")
    assert sub["status"] == "failed"
    run = db.one("select status from runs where id='run-exhaust'")
    assert run["status"] == "implementing"


def test_record_worker_activity_includes_node_id():
    """record_worker_activity includes node_id in event data when provided."""
    clear_workflow_tables()
    now = db.utcnow()
    db.execute(
        "insert into runs(id,task,brain_provider,target_path,status,created_at,updated_at) values(?,?,?,?,?,?,?)",
        ("run-act", "t", "codex", "/tmp", "implementing", now, now),
    )
    orchestrator.record_worker_activity("run-act", "implementation", {"type": "tool.started", "name": "write_file", "args": {"path": "test.py"}}, node_id="node-x")
    events = db.all_rows("select * from run_events where run_id='run-act' and event_type='agent.activity'")
    assert len(events) == 1
    data = json.loads(events[0]["data_json"])
    assert data["node_id"] == "node-x"

    orchestrator.record_worker_activity("run-act", "implementation", {"type": "tool.started", "name": "read_file", "args": {"path": "a.py"}})
    events2 = db.all_rows("select * from run_events where run_id='run-act' and event_type='agent.activity'")
    data2 = json.loads(events2[1]["data_json"])
    assert "node_id" not in data2


def test_worker_cancel_key():
    """_cancel_key composes run_id and node_id correctly."""
    assert worker._cancel_key("run-1") == "run-1"
    assert worker._cancel_key("run-1", None) == "run-1"
    assert worker._cancel_key("run-1", "node-a") == "run-1:node-a"


def test_parse_subtask_verdict():
    """parse_subtask_verdict handles structured and fallback responses."""
    sv = brain_io.parse_subtask_verdict('{"passed": true, "issues": ""}')
    assert sv.passed is True
    assert sv.issues == ""
    sv2 = brain_io.parse_subtask_verdict('{"passed": false, "issues": "missing error handling"}')
    assert sv2.passed is False
    assert sv2.issues == "missing error handling"
    sv3 = brain_io.parse_subtask_verdict("Some unstructured text with no JSON")
    assert sv3.passed is False


def test_build_subtask_verify_prompt():
    """build_subtask_verify_prompt includes title, criteria, and summary."""
    prompt = brain_io.build_subtask_verify_prompt("Add auth", "Must use JWT tokens", "Added JWT middleware")
    assert "Add auth" in prompt
    assert "JWT tokens" in prompt
    assert "JWT middleware" in prompt
    assert "passed" in prompt


# ---------------------------------------------------------------------------
# Phase 0A — Deterministic verification evidence & apply gate
# ---------------------------------------------------------------------------

def test_check_evidence_migration():
    """Migration 12 creates check_evidence table."""
    db.init_db()
    cols = {row[1] for row in db.connect().execute("pragma table_info(check_evidence)")}
    assert "run_id" in cols
    assert "exit_code" in cols
    assert "command" in cols
    assert "workspace_hash" in cols
    assert "duration_ms" in cols


def test_record_and_get_check_evidence():
    """record_check_evidence persists and get_check_evidence retrieves."""
    clear_workflow_tables()
    run_id = _seed_run()
    row_id = record_check_evidence(
        run_id=run_id, cycle=0, command="pytest",
        args=["--tb=short"], exit_code=0, output="exit=0\n2 passed",
        duration_ms=1234, workspace_hash="abc123",
    )
    assert row_id > 0
    rows = get_check_evidence(run_id)
    assert len(rows) == 1
    assert rows[0]["exit_code"] == 0
    assert rows[0]["command"] == "pytest"
    assert rows[0]["workspace_hash"] == "abc123"

    record_check_evidence(
        run_id=run_id, cycle=1, command="ruff",
        args=["check", "."], exit_code=1, output="exit=1\nerror",
        duration_ms=500, workspace_hash="def456",
    )
    all_rows = get_check_evidence(run_id)
    assert len(all_rows) == 2
    cycle_1 = get_check_evidence(run_id, cycle=1)
    assert len(cycle_1) == 1
    assert cycle_1[0]["exit_code"] == 1


def test_apply_gate_no_evidence_blocks():
    """Gate blocks when no check evidence exists, even if brain says PASS."""
    clear_workflow_tables()
    run_id = _seed_run()
    gate = evaluate_apply_gate(run_id, brain_passed=True)
    assert not gate.allowed
    assert "no_check_evidence" in gate.reasons
    assert gate.evidence_passed is False
    assert gate.brain_passed is True


def test_apply_gate_pass_with_evidence():
    """Gate allows when evidence passes and brain passes."""
    clear_workflow_tables()
    run_id = _seed_run()
    record_check_evidence(
        run_id=run_id, cycle=0, command="pytest",
        args=[], exit_code=0, output="exit=0\nok",
        duration_ms=100, workspace_hash="h1",
    )
    gate = evaluate_apply_gate(run_id, brain_passed=True)
    assert gate.allowed
    assert gate.evidence_passed is True
    assert gate.reasons == ["all_gates_passed"]


def test_apply_gate_blocks_on_check_failure():
    """Gate blocks when a check exited non-zero, even with a passing check."""
    clear_workflow_tables()
    run_id = _seed_run()
    record_check_evidence(
        run_id=run_id, cycle=0, command="ruff",
        args=["check", "."], exit_code=0, output="exit=0\nok",
        duration_ms=100, workspace_hash="h1",
    )
    record_check_evidence(
        run_id=run_id, cycle=0, command="pytest",
        args=[], exit_code=1, output="exit=1\n1 failed",
        duration_ms=200, workspace_hash="h1",
    )
    gate = evaluate_apply_gate(run_id, brain_passed=True)
    assert not gate.allowed
    assert "check_failures_present" in gate.reasons
    assert gate.pass_count == 1
    assert gate.fail_after_edit_count == 1


def test_apply_gate_blocks_brain_fail():
    """Gate blocks when brain verdict failed, even with passing checks."""
    clear_workflow_tables()
    run_id = _seed_run()
    record_check_evidence(
        run_id=run_id, cycle=0, command="pytest",
        args=[], exit_code=0, output="exit=0\nok",
        duration_ms=100, workspace_hash="h1",
    )
    gate = evaluate_apply_gate(run_id, brain_passed=False)
    assert not gate.allowed
    assert "brain_verdict_failed" in gate.reasons
    assert gate.evidence_passed is True


def test_apply_gate_blocks_no_passing_check():
    """Gate blocks when all checks failed (no exit=0)."""
    clear_workflow_tables()
    run_id = _seed_run()
    record_check_evidence(
        run_id=run_id, cycle=0, command="ruff",
        args=["check", "."], exit_code=1, output="exit=1\nerror",
        duration_ms=100, workspace_hash="h1",
    )
    gate = evaluate_apply_gate(run_id, brain_passed=True)
    assert not gate.allowed
    assert "no_passing_check" in gate.reasons
    assert "check_failures_present" in gate.reasons


def test_policy_decision_to_dict():
    """PolicyDecision.to_dict returns all fields."""
    d = PolicyDecision(
        allowed=True, reasons=["all_gates_passed"],
        evidence_passed=True, brain_passed=True,
        check_count=2, pass_count=2, fail_after_edit_count=0,
    )
    result = d.to_dict()
    assert result["allowed"] is True
    assert result["check_count"] == 2


def _seed_run() -> str:
    """Insert a minimal run row and return its id."""
    run_id = secrets.token_hex(12)
    now = db.utcnow()
    db.execute(
        "insert into runs(id,task,brain_provider,brain_model,target_path,web_research,"
        "status,baseline_hash,created_at,updated_at) values(?,?,?,?,?,?,?,?,?,?)",
        (run_id, "test task", "codex", "test-model", str(WORKSPACES), 0,
         "verifying", "testhash", now, now),
    )
    return run_id


def test_qualifying_checks_list_structure():
    """QUALIFYING_CHECKS has expected shape."""
    for check in orchestrator.QUALIFYING_CHECKS:
        assert "command" in check
        assert "args" in check
        assert isinstance(check["args"], list)
