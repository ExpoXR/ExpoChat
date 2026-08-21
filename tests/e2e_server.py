import json
import os
import sys
import tempfile
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

PORT = 31002
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
root = Path(tempfile.mkdtemp(prefix="ollma-e2e-"))
workspace = root / "workspaces" / "fixture"
workspace.mkdir(parents=True)
(workspace / "sample.py").write_text("def productivity_fixture():\n    return 'workspace context'\n")
os.environ.update(
    {
        "DATA_DIR": str(root / "data"),
        "SNAPSHOT_DIR": str(root / "snapshots"),
        "JOBS_DIR": str(root / "jobs"),
        "ALLOWED_ROOTS": str(root / "workspaces"),
        "CREDENTIAL_ENCRYPTION_KEY": "e2e-only-encryption-key",
        "SESSION_SECRET": "e2e-only-session-secret-long-value",
        "WORKER_TOKEN": "e2e-only-worker-token-long-value",
        "ADMIN_USER": "tester",
        "ADMIN_PASSWORD": "correct-horse-battery-staple",
        "ALLOW_INSECURE_PASSWORD": "true",
        "SECURE_COOKIE": "false",
        "OLLAMA_BASE_URL": f"http://127.0.0.1:{PORT}/fake-ollama",
        "WORKER_URL": f"http://127.0.0.1:{PORT}/fake-worker",
        "BRAIN_URL": f"http://127.0.0.1:{PORT}/fake-brain",
    }
)

from backend import db, plan_graph  # noqa: E402
from backend.main import app as ollma_app  # noqa: E402

app = FastAPI()


@app.get("/fake-ollama/api/tags")
def fake_tags():
    return {"models": [{"name": "test-model", "model": "test-model"}]}


@app.post("/fake-ollama/api/show")
def fake_show():
    return {"capabilities": [], "model_info": {"fixture.context_length": 8192}}


@app.get("/fake-ollama/api/version")
def fake_version():
    return {"version": "e2e"}


@app.post("/fake-ollama/api/chat")
async def fake_chat(_: Request):
    async def chunks():
        yield json.dumps({"message": {"content": "Workspace fixture answer"}, "done": False}) + "\n"
        yield json.dumps({"message": {"content": ""}, "done": True}) + "\n"

    return StreamingResponse(chunks(), media_type="application/x-ndjson")


@app.get("/fake-worker/healthz")
def fake_worker():
    return {"ok": True}


@app.get("/fake-brain/healthz")
def fake_brain():
    return {"ok": True}


db.init_db()
now = db.utcnow()
db.execute(
    "insert into runs(id,task,brain_provider,target_path,status,draft_plan,plan_state,graph_plan_hash,"
    "created_at,updated_at) values(?,?,?,?,?,?,?,?,?,?)",
    (
        "e2e-graph-run", "Graph fixture", "codex", str(workspace), "awaiting_approval",
        "Fixture plan", "refined", "fixture-hash", now, now,
    ),
)
db.insert_subtasks("e2e-graph-run", plan_graph.validate_graph({"subtasks": [
    {"node_id": "inventory", "title": "Inventory", "spec": "Inventory", "role": "research"},
    {
        "node_id": "backend", "title": "Backend", "spec": "Backend", "depends_on": ["inventory"],
        "file_globs": ["backend/**"], "complexity": "complex",
    },
    {
        "node_id": "frontend", "title": "Frontend", "spec": "Frontend", "depends_on": ["inventory"],
        "file_globs": ["public/**"],
    },
    {
        "node_id": "verify", "title": "Verify", "spec": "Verify",
        "depends_on": ["backend", "frontend"], "role": "verification",
    },
]}))
app.mount("/", ollma_app)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")
