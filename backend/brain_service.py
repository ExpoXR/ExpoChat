import hmac
import json
import os
import subprocess
import sys
import tempfile
from typing import Any, Literal

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from .config import settings
from .logging_utils import configure_logging

configure_logging()
app = FastAPI(title="Ollma Isolated Brain", docs_url=None, redoc_url=None)


class BrainRequest(BaseModel):
    provider: Literal["codex", "claude", "gemini"]
    api_key: str = Field(min_length=1, max_length=20_000)
    model: str = Field(min_length=1, max_length=200)
    prompt: str = Field(min_length=1, max_length=1_000_000)
    allow_web: bool = False
    max_output_tokens: int = Field(default=0, ge=0, le=1_000_000)
    timeout: int = Field(default=900, ge=1, le=1800)


def authorize(x_worker_token: str = Header(default="")) -> None:
    if not settings.worker_token or not hmac.compare_digest(x_worker_token, settings.worker_token):
        raise HTTPException(401, "Invalid service token")


@app.get("/healthz")
def health() -> dict[str, bool]:
    return {"ok": True}


@app.post("/execute", dependencies=[Depends(authorize)])
def execute(request: BrainRequest) -> dict[str, Any]:
    payload = request.model_dump(exclude={"timeout"})
    with tempfile.TemporaryDirectory(prefix=f"ollma-{request.provider}-", dir="/tmp") as provider_home:
        env = {
            "PATH": os.getenv("PATH", "/usr/local/bin:/usr/bin:/bin"),
            "PYTHONPATH": str(os.path.dirname(os.path.dirname(__file__))),
            "HOME": provider_home,
            "LANG": "C.UTF-8",
        }
        try:
            result = subprocess.run(
                [sys.executable, "-m", "backend.brain_runner"],
                input=json.dumps(payload),
                text=True,
                capture_output=True,
                timeout=request.timeout,
                env=env,
            )
        except subprocess.TimeoutExpired as exc:
            raise HTTPException(504, "Brain request timed out") from exc
    try:
        data = json.loads(result.stdout.strip().splitlines()[-1])
    except Exception as exc:
        raise HTTPException(502, (result.stderr or result.stdout or "Brain returned invalid output")[-4000:]) from exc
    if result.returncode != 0 or not data.get("ok"):
        raise HTTPException(502, data.get("error") or "Brain failed")
    content = str(data.get("content", ""))
    if not content.strip():
        raise HTTPException(502, "Brain returned an empty response")
    return {
        "content": content,
        "usage": data.get("usage") if isinstance(data.get("usage"), dict) else {},
    }
