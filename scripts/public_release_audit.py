#!/usr/bin/env python3
"""Fail closed when a release tree contains private or secret material."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAX_BLOB_BYTES = 5 * 1024 * 1024
FORBIDDEN_PARTS = {
    ".claude",
    ".codex",
    ".env",
    ".node",
    ".venv",
    "backups",
    "data",
    "node_modules",
    "playwright-report",
    "snapshots",
    "test-results",
}
FORBIDDEN_SUFFIXES = (".db", ".key", ".p12", ".pem", ".pfx", ".sqlite", ".sqlite3", ".tar.gz")
SAFE_VALUE_MARKERS = (
    b"change-",
    b"dummy",
    b"example",
    b"fixture",
    b"localhost",
    b"replace-",
    b"test-",
    b"your-",
)
PATTERNS = {
    "private key": re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "AWS access key": re.compile(rb"\bAKIA[0-9A-Z]{16}\b"),
    "GitHub token": re.compile(rb"\bgh[pousr]_[A-Za-z0-9_]{20,}\b"),
    "OpenAI-style secret": re.compile(rb"\bsk-[A-Za-z0-9_-]{20,}\b"),
}
ASSIGNMENT = re.compile(
    rb"(?m)^[ \t]*(?:[A-Z0-9_]*(?:API_KEY|PASSWORD|PRIVATE_KEY|SECRET|TOKEN))[ \t]*[:=][ \t]*['\"]?([^\s'\"#]+)"
)
PRIVATE_IPV4 = re.compile(
    rb"\b(?:10(?:\.\d{1,3}){3}|192\.168(?:\.\d{1,3}){2}|172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2})\b"
)
PRIVATE_MOUNT = re.compile(rb"/(?:mnt|volume\d+)/[A-Za-z0-9._/-]+")
SAFE_FIXTURE_NETWORKS = tuple(
    b".".join(str(octet).encode() for octet in address)
    for address in ((10, 0, 0, 5), (10, 20, 0, 11), (10, 20, 0, 12))
)


def git(*args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(ROOT), *args], stderr=subprocess.DEVNULL)


def tracked_paths() -> list[str]:
    return [
        item.decode()
        for item in git("ls-files", "--cached", "--others", "--exclude-standard", "-z").split(b"\0")
        if item
    ]


def secret_findings(label: str, data: bytes) -> list[str]:
    findings = [label for label, pattern in PATTERNS.items() if pattern.search(data)]
    for match in ASSIGNMENT.finditer(data):
        value = match.group(1).lower()
        if value.startswith((b"$", b"<", b"{")) or any(marker in value for marker in SAFE_VALUE_MARKERS):
            continue
        findings.append("credential-like assignment")
        break
    source_path = label.removeprefix("history:")
    topology_data = data
    if source_path.startswith("tests/") or source_path == "playwright.config.js":
        for fixture in SAFE_FIXTURE_NETWORKS:
            topology_data = topology_data.replace(fixture, b"")
    if PRIVATE_IPV4.search(topology_data) or PRIVATE_MOUNT.search(topology_data):
        findings.append("private infrastructure reference")
    return findings


def inspect_file(label: str, data: bytes, issues: list[str]) -> None:
    if len(data) > MAX_BLOB_BYTES:
        issues.append(f"oversized tracked blob: {label}")
        return
    if b"\0" in data[:4096]:
        return
    for finding in secret_findings(label, data):
        issues.append(f"{finding}: {label}")


def audit_tree(issues: list[str]) -> None:
    for name in tracked_paths():
        path = Path(name)
        file_path = ROOT / path
        if not file_path.is_file():
            continue
        forbidden_part = next((part for part in path.parts if part in FORBIDDEN_PARTS), None)
        if forbidden_part and name != ".env.example":
            issues.append(f"private runtime path tracked: {name}")
            continue
        if name.lower().endswith(FORBIDDEN_SUFFIXES):
            issues.append(f"private artifact tracked: {name}")
            continue
        inspect_file(name, file_path.read_bytes(), issues)


def audit_history(issues: list[str]) -> None:
    objects = git("rev-list", "--objects", "--all").decode().splitlines()
    seen: set[str] = set()
    for entry in objects:
        object_id, _, name = entry.partition(" ")
        if not name or object_id in seen:
            continue
        seen.add(object_id)
        if name != ".env.example" and any(part in FORBIDDEN_PARTS for part in Path(name).parts):
            issues.append(f"private runtime path in history: {name}")
            continue
        if name.lower().endswith(FORBIDDEN_SUFFIXES):
            issues.append(f"private artifact in history: {name}")
            continue
        try:
            data = git("cat-file", "blob", object_id)
        except subprocess.CalledProcessError:
            continue
        inspect_file(f"history:{name}", data, issues)


def main() -> int:
    issues: list[str] = []
    audit_tree(issues)
    if "--history" in sys.argv:
        audit_history(issues)
    if issues:
        print("Public release audit failed:", file=sys.stderr)
        for issue in sorted(set(issues)):
            print(f"- {issue}", file=sys.stderr)
        return 1
    print("Public release audit passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
