import os
import re
import subprocess
from pathlib import Path
from typing import Any

from .config import settings

MAX_FILE_BYTES = 2_000_000
MAX_TOOL_OUTPUT = 40_000
SAFE_COMMANDS = {"python", "python3", "pytest", "node", "npm", "npx", "git", "rg", "grep", "ruff"}
EXCLUDED = {".git", "node_modules", "__pycache__", ".next", ".cache", "dist", "build", "vendor"}


def safe_path(root: Path, value: str = ".") -> Path:
    root = root.resolve()
    candidate = (root / value).resolve(strict=False)
    if candidate != root and root not in candidate.parents:
        raise ValueError("Path escapes staged workspace")
    return candidate


def list_files(root: Path, path: str = ".", limit: int = 300) -> str:
    base = safe_path(root, path)
    if not base.exists():
        return "Path not found"
    rows: list[str] = []
    if base.is_file():
        return str(base.relative_to(root))
    for current, dirs, files in os.walk(base):
        dirs[:] = sorted(d for d in dirs if d not in EXCLUDED)
        for name in sorted(files):
            item = Path(current) / name
            if item.is_symlink():
                continue
            rows.append(str(item.relative_to(root)))
            if len(rows) >= min(limit, 1000):
                rows.append("...truncated")
                return "\n".join(rows)
    return "\n".join(rows)


def read_file(root: Path, path: str, start_line: int = 1, end_line: int = 400) -> str:
    target = safe_path(root, path)
    if not target.is_file():
        return "File not found"
    if target.stat().st_size > MAX_FILE_BYTES:
        return "File exceeds 2 MB limit"
    lines = target.read_text(errors="replace").splitlines()
    start = max(1, start_line)
    end = min(len(lines), max(start, end_line), start + 799)
    return "\n".join(f"{i}: {lines[i - 1]}" for i in range(start, end + 1))


def search_files(root: Path, pattern: str, path: str = ".") -> str:
    base = safe_path(root, path)
    try:
        regex = re.compile(pattern)
    except re.error as exc:
        return f"Invalid regex: {exc}"
    hits: list[str] = []
    candidates: list[Path] = [base] if base.is_file() else []
    if base.is_dir():
        for current, dirs, files in os.walk(base):
            dirs[:] = [name for name in dirs if name not in EXCLUDED]
            candidates.extend(Path(current) / name for name in files)
    for candidate in candidates:
        try:
            if candidate.is_symlink() or not candidate.is_file() or candidate.stat().st_size > MAX_FILE_BYTES:
                continue
            for number, line in enumerate(candidate.read_text(errors="replace").splitlines(), 1):
                if regex.search(line):
                    hits.append(f"{candidate.relative_to(root)}:{number}:{line[:500]}")
                    if len(hits) >= 300:
                        return "\n".join(hits + ["...truncated"])
        except OSError:
            continue
    return "\n".join(hits)


def search_text(root: Path, query: str, cursor: int = 0, limit: int = 50) -> dict[str, Any]:
    root = root.resolve()
    limit = min(max(limit, 1), 100)
    cursor = max(cursor, 0)
    command = [
        "rg",
        "--fixed-strings",
        "--line-number",
        "--no-heading",
        "--color=never",
        "--max-filesize=2M",
    ]
    for name in sorted(EXCLUDED):
        command.extend(["--glob", f"!{name}/**"])
    command.extend(["--", query, str(root)])
    try:
        process = subprocess.Popen(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={"PATH": os.getenv("PATH", "")},
        )
        lines = []
        assert process.stdout is not None
        for line in process.stdout:
            lines.append(line.rstrip("\n"))
            if len(lines) >= cursor + limit + 1:
                process.terminate()
                break
        process.wait(timeout=5)
        if process.returncode not in {0, 1, -15}:
            assert process.stderr is not None
            raise RuntimeError(process.stderr.read(1000))
        selected = lines[cursor : cursor + limit]
        rows = []
        for hit in selected:
            file_name, number, text = hit.split(":", 2)
            rows.append({"file": file_name, "line": int(number), "text": text[:500]})
        next_cursor = cursor + len(selected) if len(lines) > cursor + len(selected) else None
        return {"results": rows, "next_cursor": next_cursor, "truncated": next_cursor is not None}
    except (FileNotFoundError, RuntimeError, subprocess.TimeoutExpired, ValueError):
        raw = search_files(root, re.escape(query)).splitlines()
        selected = raw[cursor : cursor + limit]
        rows = []
        for hit in selected:
            file_name, number, text = hit.split(":", 2)
            rows.append({"file": str(root / file_name), "line": int(number), "text": text})
        next_cursor = cursor + len(selected) if cursor + len(selected) < len(raw) else None
        return {"results": rows, "next_cursor": next_cursor, "truncated": next_cursor is not None}


def write_file(root: Path, path: str, content: str) -> str:
    target = safe_path(root, path)
    if len(content.encode()) > MAX_FILE_BYTES:
        return "Rejected: content exceeds 2 MB"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)
    return f"Wrote {target.relative_to(root)}"


def replace_text(root: Path, path: str, old: str, new: str, replace_all: bool = False) -> str:
    target = safe_path(root, path)
    if not target.is_file():
        return "File not found"
    content = target.read_text(errors="replace")
    count = content.count(old)
    if count == 0:
        return "Text not found"
    if count > 1 and not replace_all:
        return f"Rejected: text occurs {count} times; provide more context or set replace_all"
    changed = content.replace(old, new) if replace_all else content.replace(old, new, 1)
    target.write_text(changed)
    return f"Updated {target.relative_to(root)} ({count if replace_all else 1} replacement(s))"


def delete_file(root: Path, path: str) -> str:
    target = safe_path(root, path)
    if not target.is_file() and not target.is_symlink():
        return "File not found or not a regular file"
    target.unlink()
    return f"Deleted {target.relative_to(root)}"


def run_check(root: Path, command: str, args: list[str] | None = None, timeout: int = 120) -> str:
    executable = Path(command).name
    if executable not in SAFE_COMMANDS or command != executable:
        return "Rejected command"
    args = [str(item) for item in (args or [])][:40]
    for arg in args:
        if "\x00" in arg or len(arg) > 1000:
            return "Rejected argument"
        if os.path.isabs(arg):
            try:
                safe_path(root, arg)
            except ValueError:
                return "Rejected path outside staged workspace"
    if executable in {"python", "python3", "node"} and any(arg in {"-c", "-e", "--eval"} for arg in args):
        return "Rejected inline program"
    if executable == "git" and (not args or args[0] not in {"status", "diff", "log", "show", "rev-parse", "ls-files", "grep"}):
        return "Rejected mutating git command"
    if executable == "npm" and (not args or args[0] not in {"test", "run", "exec"}):
        return "Rejected npm command"
    if executable == "npx" and (not args or args[0] != "--no-install"):
        return "Rejected npx command; use --no-install"
    env = {
        "PATH": os.getenv("PATH", "/usr/local/bin:/usr/bin:/bin"),
        "HOME": "/tmp/ollma-worker-home",
        "LANG": "C.UTF-8",
        "PYTHONDONTWRITEBYTECODE": "1",
        "CI": "1",
        "npm_config_yes": "false",
    }
    Path(env["HOME"]).mkdir(parents=True, exist_ok=True)
    try:
        result = subprocess.run(
            [executable, *args], cwd=root, env=env, text=True, capture_output=True,
            timeout=min(max(timeout, 1), settings.command_timeout),
        )
        return f"exit={result.returncode}\n{result.stdout}{result.stderr}"[:MAX_TOOL_OUTPUT]
    except subprocess.TimeoutExpired:
        return "Command timed out"
    except FileNotFoundError:
        return f"Command unavailable: {executable}"


TOOL_DEFS = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files in workspace.",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read numbered lines from a text file.",
            "parameters": {
                "type": "object",
                "required": ["path"],
                "properties": {"path": {"type": "string"}, "start_line": {"type": "integer"}, "end_line": {"type": "integer"}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "Regex search text files.",
            "parameters": {"type": "object", "required": ["pattern"], "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Create or replace a text file.",
            "parameters": {"type": "object", "required": ["path", "content"], "properties": {"path": {"type": "string"}, "content": {"type": "string"}}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "replace_text",
            "description": "Precisely replace text in a file.",
            "parameters": {
                "type": "object",
                "required": ["path", "old", "new"],
                "properties": {"path": {"type": "string"}, "old": {"type": "string"}, "new": {"type": "string"}, "replace_all": {"type": "boolean"}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_file",
            "description": "Delete one regular file.",
            "parameters": {"type": "object", "required": ["path"], "properties": {"path": {"type": "string"}}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_check",
            "description": "Run an allowlisted check without a shell.",
            "parameters": {
                "type": "object",
                "required": ["command"],
                "properties": {"command": {"type": "string"}, "args": {"type": "array", "items": {"type": "string"}}, "timeout": {"type": "integer"}},
            },
        },
    },
]


def execute_tool(root: Path, name: str, args: dict[str, Any], writable: bool) -> str:
    if name == "list_files":
        return list_files(root, args.get("path", "."), int(args.get("limit", 300)))
    if name == "read_file":
        return read_file(root, args["path"], int(args.get("start_line", 1)), int(args.get("end_line", 400)))
    if name == "search_files":
        return search_files(root, args["pattern"], args.get("path", "."))
    if name == "run_check":
        return run_check(root, args["command"], args.get("args"), int(args.get("timeout", 120)))
    if not writable:
        return "Rejected: worker is read-only"
    if name == "write_file":
        return write_file(root, args["path"], args["content"])
    if name == "replace_text":
        return replace_text(root, args["path"], args["old"], args["new"], bool(args.get("replace_all", False)))
    if name == "delete_file":
        return delete_file(root, args["path"])
    return "Unknown tool"
