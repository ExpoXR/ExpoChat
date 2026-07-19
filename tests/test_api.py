import asyncio

import httpx

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
                storage = await client.get("/api/maintenance/storage")
                assert storage.status_code == 200
                assert "orphans" in storage.json()

    asyncio.run(run())
