import asyncio
import json
import secrets

import httpx

from backend import brain_service, db, worker, workspace
from backend.main import app


def test_health_and_csrf_protection():
    async def run():
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                assert (await client.get("/livez")).status_code == 200
                assert "Ollma UI" in (await client.get("/")).text
                assert (await client.get("/internal/ollama/api/version")).status_code == 401
                login = await client.post("/api/auth/login", json={"username": "tester", "password": "correct-horse-battery-staple"})
                assert login.status_code == 200
                csrf = login.json()["csrf"]
                assert (await client.post("/api/agents/discover", json={})).status_code == 403
                response = await client.put(
                    "/api/brains",
                    headers={"X-CSRF-Token": csrf},
                    json={"provider": "codex", "model": "gpt-test", "enabled": False},
                )
                assert response.status_code == 200
                linked = await client.put(
                    "/api/brains",
                    headers={"X-CSRF-Token": csrf},
                    json={"provider": "codex", "model": "gpt-5.6-terra", "api_key": "test-api-key", "enabled": True},
                )
                assert linked.json()["linked"] is True
                disconnected = await client.put(
                    "/api/brains",
                    headers={"X-CSRF-Token": csrf},
                    json={"provider": "codex", "model": "gpt-5.6-terra", "enabled": False},
                )
                assert disconnected.status_code == 200
                missing_key = await client.put(
                    "/api/brains",
                    headers={"X-CSRF-Token": csrf},
                    json={"provider": "codex", "model": "gpt-5.6-terra", "enabled": True},
                )
                assert missing_key.status_code == 400
                brain_rows = (await client.get("/api/brains")).json()["brains"]
                assert next(row for row in brain_rows if row["provider"] == "codex")["model"] == "gpt-5.6-terra"
                for title in ("First", "Second"):
                    created = await client.post(
                        "/api/chats",
                        headers={"X-CSRF-Token": csrf},
                        json={"title": title, "model": "test-model"},
                    )
                    assert created.status_code == 200
                page = await client.get("/api/chats?limit=1")
                assert len(page.json()["chats"]) == 1
                assert page.json()["next_cursor"] == 1
                selected = page.json()["chats"][0]
                detail = await client.get(f"/api/chats/{selected['id']}/messages")
                assert detail.json()["chat"]["title"] == selected["title"]
                storage = await client.get("/api/maintenance/storage")
                assert storage.status_code == 200
                assert "orphans" in storage.json()

                target = workspace.settings.allowed_roots[0] / "protected-snapshot"
                target.mkdir(exist_ok=True)
                (target / "value.txt").write_text("before")
                snapshot = workspace.create_snapshot(target)
                run_id = secrets.token_hex(12)
                now = db.utcnow()
                db.execute(
                    "insert into runs(id,task,brain_provider,target_path,status,snapshot_id,created_at,updated_at) values(?,?,?,?,?,?,?,?)",
                    (run_id, "Protect snapshot", "codex", str(target), "implementing", snapshot["id"], now, now),
                )
                listed = await client.get("/api/snapshots")
                protected = next(item for item in listed.json()["snapshots"] if item["id"] == snapshot["id"])
                assert protected["protected"] is True
                assert (await client.post(
                    f"/api/snapshots/{snapshot['id']}/restore",
                    headers={"X-CSRF-Token": csrf},
                    json={},
                )).status_code == 409
                assert (await client.delete(
                    f"/api/snapshots/{snapshot['id']}",
                    headers={"X-CSRF-Token": csrf},
                )).status_code == 409

    asyncio.run(run())


def test_fs_operations_and_chat_management():
    async def run():
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                login = await client.post("/api/auth/login", json={"username": "tester", "password": "correct-horse-battery-staple"})
                csrf = login.json()["csrf"]
                h = {"X-CSRF-Token": csrf}
                root = workspace.settings.allowed_roots[0]
                folder = root / "fsops"

                assert (await client.post("/api/fs/folder", headers=h, json={"path": str(folder)})).status_code == 200
                assert folder.is_dir()

                src = folder / "a.txt"
                assert (await client.put("/api/file", headers=h, json={"path": str(src), "content": "hello"})).status_code == 200

                copied = await client.post("/api/fs/copy", headers=h, json={"src": str(src), "dest": str(folder / "b.txt")})
                assert copied.status_code == 200
                assert (folder / "b.txt").read_text() == "hello"
                # copy without overwrite onto an existing path is rejected
                assert (await client.post("/api/fs/copy", headers=h, json={"src": str(src), "dest": str(folder / "b.txt")})).status_code == 409

                renamed = await client.post("/api/fs/rename", headers=h, json={"path": str(folder / "b.txt"), "new_path": str(folder / "c.txt")})
                assert renamed.status_code == 200
                assert (folder / "c.txt").exists() and not (folder / "b.txt").exists()

                # deletion snapshots first, so rollback stays possible
                before = len((await client.get("/api/snapshots")).json()["snapshots"])
                deleted = await client.post("/api/fs/delete", headers=h, json={"path": str(folder / "c.txt")})
                assert deleted.status_code == 200 and deleted.json()["snapshot"]
                assert not (folder / "c.txt").exists()
                after = len((await client.get("/api/snapshots")).json()["snapshots"])
                assert after == before + 1

                # sandbox guards
                assert (await client.post("/api/fs/delete", headers=h, json={"path": "/etc/hosts"})).status_code == 403
                assert (await client.post("/api/fs/delete", headers=h, json={"path": str(root)})).status_code == 400

                # chat management
                c1 = (await client.post("/api/chats", headers=h, json={"title": "Alpha", "model": "m"})).json()["chat"]
                c2 = (await client.post("/api/chats", headers=h, json={"title": "Beta", "model": "m"})).json()["chat"]
                renamed_chat = await client.patch(f"/api/chats/{c1['id']}", headers=h, json={"title": "Alpha2"})
                assert renamed_chat.status_code == 200 and renamed_chat.json()["title"] == "Alpha2"
                assert (await client.patch(f"/api/chats/{c1['id']}", headers=h, json={"pinned": True})).status_code == 200
                assert (await client.get("/api/chats")).json()["chats"][0]["id"] == c1["id"]
                dup = (await client.post(f"/api/chats/{c2['id']}/duplicate", headers=h, json={})).json()["chat"]
                assert dup["title"] == "Beta (copy)" and dup["id"] != c2["id"]
                assert (await client.delete(f"/api/chats/{c2['id']}", headers=h)).status_code == 200
                assert (await client.get(f"/api/chats/{c2['id']}/messages")).status_code == 404

    asyncio.run(run())


def test_worker_cancellation_interrupts_active_task(monkeypatch):
    async def run():
        run_id = "d" * 24
        (worker.settings.jobs_dir / run_id / "workspace").mkdir(parents=True, exist_ok=True)
        started = asyncio.Event()

        async def slow_agent(*_):
            started.set()
            await asyncio.sleep(60)
            return {"ok": True}

        monkeypatch.setattr(worker, "agent_loop", slow_agent)
        transport = httpx.ASGITransport(app=worker.app)
        headers = {"X-Worker-Token": worker.settings.worker_token}
        payload = {
            "run_id": run_id,
            "model": "test-model",
            "mode": "research",
            "task": "Wait for cancellation",
        }
        async with httpx.AsyncClient(transport=transport, base_url="http://worker") as client:
            request = asyncio.create_task(client.post("/execute", headers=headers, json=payload))
            await asyncio.wait_for(started.wait(), timeout=1)
            cancelled = await client.post(f"/cancel/{run_id}", headers=headers)
            response = await asyncio.wait_for(request, timeout=1)
        assert cancelled.status_code == 200
        assert cancelled.json()["active"] is True
        assert response.json()["cancelled"] is True

    asyncio.run(run())


def test_worker_stream_emits_tool_activity_before_result(monkeypatch):
    async def run():
        run_id = "c" * 24
        (worker.settings.jobs_dir / run_id / "workspace").mkdir(parents=True, exist_ok=True)

        async def fake_agent(_request, _root, _cancelled=None, emit=None):
            await emit({"type": "tool.started", "turn": 1, "name": "write_file", "args": {"path": "app.py"}})
            await emit({"type": "tool.completed", "turn": 1, "name": "write_file", "args": {"path": "app.py"}, "result": "Wrote app.py"})
            return {"ok": True, "content": "done", "events": []}

        monkeypatch.setattr(worker, "agent_loop", fake_agent)
        transport = httpx.ASGITransport(app=worker.app)
        headers = {"X-Worker-Token": worker.settings.worker_token}
        payload = {"run_id": run_id, "model": "test-model", "mode": "implementation", "task": "Edit app"}
        async with httpx.AsyncClient(transport=transport, base_url="http://worker") as client:
            response = await client.post("/execute/stream", headers=headers, json=payload)
        items = [json.loads(line) for line in response.text.splitlines()]
        assert [item["type"] for item in items] == ["tool.started", "tool.completed", "result"]
        assert items[-1]["result"]["ok"] is True

    asyncio.run(run())


def test_worker_console_check_is_isolated_and_reports_exit_status():
    async def run():
        run_id = "a" * 24
        stage = worker.settings.jobs_dir / run_id / "workspace"
        stage.mkdir(parents=True, exist_ok=True)
        (stage / "value.txt").write_text("needle\n")
        transport = httpx.ASGITransport(app=worker.app)
        headers = {"X-Worker-Token": worker.settings.worker_token}
        async with httpx.AsyncClient(transport=transport, base_url="http://worker") as client:
            passed = await client.post(
                "/check",
                headers=headers,
                json={"run_id": run_id, "command": "grep", "args": ["needle", "value.txt"]},
            )
            rejected = await client.post(
                "/check",
                headers=headers,
                json={"run_id": run_id, "command": "grep", "args": ["needle", "../outside.txt"]},
            )
        assert passed.status_code == 200
        assert passed.json()["ok"] is True
        assert rejected.json() == {"ok": False, "content": "Rejected path traversal"}

    asyncio.run(run())


def test_brain_service_requires_auth_and_forwards_provider_request(monkeypatch):
    captured = {}

    def fake_run(*_args, **kwargs):
        captured.update(kwargs)
        return type("Result", (), {"returncode": 0, "stdout": '{"ok":true,"content":"OK","usage":{"total_tokens":3}}\n', "stderr": ""})()

    monkeypatch.setattr(brain_service.subprocess, "run", fake_run)

    async def run():
        transport = httpx.ASGITransport(app=brain_service.app)
        payload = {"provider": "codex", "api_key": "secret", "model": "gpt-test", "prompt": "ping", "allow_web": True}
        async with httpx.AsyncClient(transport=transport, base_url="http://brain") as client:
            assert (await client.post("/execute", json=payload)).status_code == 401
            response = await client.post(
                "/execute",
                headers={"X-Worker-Token": brain_service.settings.worker_token},
                json=payload,
            )
        assert response.status_code == 200
        assert response.json() == {"content": "OK", "usage": {"total_tokens": 3}}

    asyncio.run(run())
    forwarded = json.loads(captured["input"])
    assert forwarded["allow_web"] is True
    assert forwarded["api_key"] == "secret"
    assert captured["env"]["HOME"].startswith("/tmp/ollma-codex-")
