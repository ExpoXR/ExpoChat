import os
import sys
import tempfile
from pathlib import Path

TEST_ROOT = Path(tempfile.mkdtemp(prefix="ollma-tests-"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
(TEST_ROOT / "workspaces").mkdir()
os.environ.update({
    "OLLMA_TEST_ROOT": str(TEST_ROOT),
    "DATA_DIR": str(TEST_ROOT / "data"),
    "SNAPSHOT_DIR": str(TEST_ROOT / "snapshots"),
    "JOBS_DIR": str(TEST_ROOT / "jobs"),
    "ALLOWED_ROOTS": str(TEST_ROOT / "workspaces"),
    "CREDENTIAL_ENCRYPTION_KEY": "test-only-encryption-key",
    "SESSION_SECRET": "test-only-session-secret-long-value",
    "WORKER_TOKEN": "test-only-worker-token-long-value",
    "ADMIN_USER": "tester",
    "ADMIN_PASSWORD": "correct-horse-battery-staple",
    "ALLOW_INSECURE_PASSWORD": "true",
    "SECURE_COOKIE": "false",
})
