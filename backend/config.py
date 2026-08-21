import os
from dataclasses import dataclass, field
from pathlib import Path


def _paths(value: str) -> list[Path]:
    return [Path(item.strip()).resolve() for item in value.split(",") if item.strip()]


def _int(name: str, default: int, minimum: int = 0) -> int:
    try:
        return max(minimum, int(os.getenv(name, str(default))))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("DATA_DIR", "/data")))
    snapshot_dir: Path = field(default_factory=lambda: Path(os.getenv("SNAPSHOT_DIR", "/snapshots")))
    jobs_dir: Path = field(default_factory=lambda: Path(os.getenv("JOBS_DIR", "/jobs")))
    allowed_roots: list[Path] = field(
        default_factory=lambda: _paths(os.getenv("ALLOWED_ROOTS", "/workspace"))
    )
    public_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1] / "public")
    admin_user: str = field(default_factory=lambda: os.getenv("ADMIN_USER", "admin"))
    admin_password: str = field(default_factory=lambda: os.getenv("ADMIN_PASSWORD", "change-me-now"))
    admin_password_hash: str = field(default_factory=lambda: os.getenv("ADMIN_PASSWORD_HASH", ""))
    # Dev-only escape hatch: allow a plaintext/md5 ADMIN_PASSWORD when no argon2
    # hash is set. Off by default so production must use ADMIN_PASSWORD_HASH.
    allow_insecure_password: bool = field(
        default_factory=lambda: os.getenv("ALLOW_INSECURE_PASSWORD", "false").lower() == "true"
    )
    session_secret: str = field(default_factory=lambda: os.getenv("SESSION_SECRET", "change-this-session-secret"))
    credential_key: str = field(default_factory=lambda: os.getenv("CREDENTIAL_ENCRYPTION_KEY", ""))
    ollama_url: str = field(default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/"))
    worker_url: str = field(default_factory=lambda: os.getenv("WORKER_URL", "http://ollma-worker:8090").rstrip("/"))
    brain_url: str = field(default_factory=lambda: os.getenv("BRAIN_URL", "http://ollma-brain:8091").rstrip("/"))
    worker_token: str = field(default_factory=lambda: os.getenv("WORKER_TOKEN", "change-worker-token"))
    command_timeout: int = field(default_factory=lambda: _int("COMMAND_TIMEOUT", 120, 1))
    allowed_origins: set[str] = field(
        default_factory=lambda: {x.strip() for x in os.getenv("ALLOWED_ORIGINS", "").split(",") if x.strip()}
    )
    # Force-secure override. Default off; when unset, security.py still auto-detects HTTPS
    # from the request scheme / X-Forwarded-Proto, so cookies stay Secure behind a TLS proxy.
    secure_cookie: bool = field(default_factory=lambda: os.getenv("SECURE_COOKIE", "false").lower() == "true")
    snapshot_retention_days: int = field(default_factory=lambda: _int("SNAPSHOT_RETENTION_DAYS", 30, 1))
    snapshot_max_bytes: int = field(default_factory=lambda: _int("SNAPSHOT_MAX_BYTES", 20 * 1024**3, 1))
    snapshot_reserve_bytes: int = field(default_factory=lambda: _int("SNAPSHOT_RESERVE_BYTES", 2 * 1024**3, 0))
    orphan_grace_hours: int = field(default_factory=lambda: _int("ORPHAN_GRACE_HOURS", 24, 1))
    runner_concurrency: int = field(default_factory=lambda: _int("RUNNER_CONCURRENCY", 1, 1))
    # Max subtasks a single run executes in parallel. Default 1 = serialized on one GPU
    # (parallel Ollama calls would just queue). Raise once the host serves models concurrently.
    worker_pool_size: int = field(default_factory=lambda: _int("WORKER_POOL_SIZE", 1, 1))
    chat_context_bytes: int = field(default_factory=lambda: _int("CHAT_CONTEXT_BYTES", 120_000, 10_000))
    openai_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    openai_model: str = field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-5.6-sol"))
    claude_key: str = field(default_factory=lambda: os.getenv("CLAUDE_API_KEY", ""))
    claude_model: str = field(default_factory=lambda: os.getenv("CLAUDE_MODEL", "claude-sonnet-5"))
    gemini_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    gemini_model: str = field(default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-2.5-pro"))

    def environment_key(self, provider: str) -> str:
        return {"codex": self.openai_key, "claude": self.claude_key, "gemini": self.gemini_key}.get(provider, "")

    @property
    def db_path(self) -> Path:
        return self.data_dir / "ollma.sqlite3"

settings = Settings()
