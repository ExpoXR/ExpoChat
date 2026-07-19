import base64
import hashlib
import hmac
import logging
import secrets
import threading
import time
from collections import defaultdict, deque
from typing import Any

from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
from cryptography.fernet import Fernet, InvalidToken
from fastapi import HTTPException, Request, Response, status

from . import db
from .config import settings

log = logging.getLogger("ollma.security")
_hasher = PasswordHasher()
_rate_lock = threading.Lock()
_rate_windows: dict[str, deque[float]] = defaultdict(deque)


def _fernet() -> Fernet:
    if not settings.credential_key:
        raise HTTPException(503, "CREDENTIAL_ENCRYPTION_KEY is required before storing API keys")
    raw = settings.credential_key.encode()
    try:
        return Fernet(raw)
    except ValueError:
        derived = base64.urlsafe_b64encode(hashlib.sha256(raw).digest())
        return Fernet(derived)


def encrypt_secret(value: str) -> str:
    return _fernet().encrypt(value.encode()).decode()


def decrypt_secret(value: str | None) -> str:
    if not value:
        return ""
    try:
        return _fernet().decrypt(value.encode()).decode()
    except InvalidToken as exc:
        raise HTTPException(503, "Stored credential cannot be decrypted") from exc


def verify_password(candidate: str) -> bool:
    stored_hash = settings.admin_password_hash
    if stored_hash:
        try:
            return _hasher.verify(stored_hash, candidate)
        except (VerifyMismatchError, ValueError):
            return False
    stored = settings.admin_password
    if stored.startswith("md5:"):
        log.warning("legacy_md5_admin_password_configured")
        digest = hashlib.md5(candidate.encode(), usedforsecurity=False).hexdigest()
        return hmac.compare_digest(digest, stored[4:])
    if stored in {"change-me-now", ""}:
        log.error("unsafe_default_admin_password_configured")
    return hmac.compare_digest(candidate, stored)


def rate_limit(key: str, limit: int, period: int = 60) -> None:
    now = time.time()
    with _rate_lock:
        window = _rate_windows[key]
        while window and window[0] <= now - period:
            window.popleft()
        if len(window) >= limit:
            raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "Rate limit exceeded")
        window.append(now)


def create_session(response: Response, request: Request) -> dict[str, str]:
    session_id = secrets.token_urlsafe(32)
    csrf = secrets.token_urlsafe(24)
    now = time.time()
    absolute_expiry = now + 7 * 86400
    db.execute(
        "insert into sessions(id,user,csrf,created_at,last_seen,expires_at) values(?,?,?,?,?,?)",
        (session_id, settings.admin_user, csrf, now, now, absolute_expiry),
    )
    forwarded_proto = request.headers.get("x-forwarded-proto", "").split(",")[0].strip()
    secure = settings.secure_cookie or forwarded_proto == "https" or request.url.scheme == "https"
    response.set_cookie(
        "ollma_session",
        session_id,
        httponly=True,
        secure=secure,
        samesite="strict",
        max_age=7 * 86400,
        path="/",
    )
    return {"user": settings.admin_user, "csrf": csrf}


def destroy_session(request: Request, response: Response) -> None:
    session_id = request.cookies.get("ollma_session", "")
    if session_id:
        db.execute("delete from sessions where id=?", (session_id,))
    response.delete_cookie("ollma_session", path="/")


def session_for(request: Request, touch: bool = True) -> dict[str, Any] | None:
    session_id = request.cookies.get("ollma_session", "")
    if not session_id:
        return None
    row = db.one("select * from sessions where id=?", (session_id,))
    now = time.time()
    if not row or row["expires_at"] <= now or row["last_seen"] <= now - 12 * 3600:
        if row:
            db.execute("delete from sessions where id=?", (session_id,))
        return None
    if touch and row["last_seen"] <= now - 60:
        db.execute("update sessions set last_seen=? where id=?", (now, session_id))
        row["last_seen"] = now
    return row


def check_origin(request: Request) -> None:
    origin = request.headers.get("origin")
    if not origin:
        return
    expected = f"{request.url.scheme}://{request.headers.get('host', '')}"
    forwarded_proto = request.headers.get("x-forwarded-proto", "").split(",")[0].strip()
    if forwarded_proto:
        expected = f"{forwarded_proto}://{request.headers.get('host', '')}"
    if origin != expected and origin not in settings.allowed_origins:
        raise HTTPException(403, "Origin not allowed")


def require_user(request: Request) -> dict[str, Any]:
    row = session_for(request)
    if not row:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Unauthorized")
    if request.method not in {"GET", "HEAD", "OPTIONS"}:
        check_origin(request)
        token = request.headers.get("x-csrf-token", "")
        if not token or not hmac.compare_digest(token, row["csrf"]):
            raise HTTPException(403, "CSRF validation failed")
    rate_limit(f"api:{row['id']}", 300)
    return row
