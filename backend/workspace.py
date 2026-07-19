import contextlib
import hashlib
import json
import os
import shutil
import tarfile
import tempfile
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from fastapi import HTTPException

from . import db
from .config import settings

EXCLUDED_DIRS = {".git", "node_modules", "__pycache__", ".next", ".cache", "dist", "build", "vendor"}


def _copy_ignore(directory: str, names: list[str]) -> list[str]:
    base = Path(directory)
    return [name for name in names if name in EXCLUDED_DIRS or (base / name).is_symlink()]


def allowed_path(value: str | Path, must_exist: bool = True) -> Path:
    candidate = Path(value).expanduser().resolve(strict=False)
    if not any(candidate == root or root in candidate.parents for root in settings.allowed_roots):
        raise HTTPException(403, "Path is outside ALLOWED_ROOTS")
    if must_exist and not candidate.exists():
        raise HTTPException(404, "Path does not exist")
    return candidate


def iter_files(root: Path):
    if root.is_file():
        yield root
        return
    for base, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d not in EXCLUDED_DIRS)
        for name in sorted(files):
            path = Path(base) / name
            with contextlib.suppress(OSError):
                if path.is_file() and not path.is_symlink():
                    yield path


def workspace_manifest(root: Path) -> dict[str, dict[str, Any]]:
    base = root.parent if root.is_file() else root
    result: dict[str, dict[str, Any]] = {}
    for path in iter_files(root):
        rel = path.name if root.is_file() else str(path.relative_to(base))
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        stat = path.stat()
        result[rel] = {"sha256": digest.hexdigest(), "size": stat.st_size, "mode": stat.st_mode & 0o777}
    return result


def manifest_hash(root: Path) -> str:
    data = json.dumps(workspace_manifest(root), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()


def stage_workspace(run_id: str, source: Path, destination_name: str = "workspace") -> Path:
    run_dir = settings.jobs_dir / run_id
    destination = run_dir / destination_name
    if destination.exists():
        shutil.rmtree(destination)
    run_dir.mkdir(parents=True, exist_ok=True)
    if source.is_file():
        destination.mkdir(parents=True)
        shutil.copy2(source, destination / source.name)
    else:
        shutil.copytree(
            source,
            destination,
            symlinks=False,
            ignore=_copy_ignore,
        )
    return destination


def _same_path(left: Path, right: Path) -> bool:
    try:
        return left.exists() and right.exists() and os.path.samefile(left, right)
    except OSError:
        return False


def _snapshot_entries(path: Path) -> list[tuple[Path, str]]:
    if path.is_file():
        return [(path, path.name)]
    entries: list[tuple[Path, str]] = [(path, path.name)]
    runtime_dirs = (settings.data_dir, settings.jobs_dir, settings.snapshot_dir)
    for base, dirs, files in os.walk(path):
        base_path = Path(base)
        kept_dirs = []
        for name in sorted(dirs):
            item = base_path / name
            if name in EXCLUDED_DIRS or item.is_symlink() or any(_same_path(item, runtime) for runtime in runtime_dirs):
                continue
            kept_dirs.append(name)
            entries.append((item, str(Path(path.name) / item.relative_to(path))))
        dirs[:] = kept_dirs
        for name in sorted(files):
            item = base_path / name
            if item.is_symlink():
                continue
            with contextlib.suppress(OSError):
                if item.is_file():
                    entries.append((item, str(Path(path.name) / item.relative_to(path))))
    return entries


def create_snapshot(path: Path, chat_id: str | None = None) -> dict[str, Any]:
    import secrets

    path = allowed_path(path)
    snap_id = f"snap-{int(time.time())}-{secrets.token_hex(4)}"
    archive_name = f"{snap_id}.tar.gz"
    archive = settings.snapshot_dir / archive_name
    partial = settings.snapshot_dir / f".{archive_name}.part"
    entries = _snapshot_entries(path)
    source_bytes = sum(item.stat().st_size for item, _ in entries if item.is_file())
    if source_bytes > settings.snapshot_max_bytes:
        raise RuntimeError(
            f"Snapshot source exceeds limit ({source_bytes} > {settings.snapshot_max_bytes} bytes)"
        )
    free = shutil.disk_usage(settings.snapshot_dir).free
    required = source_bytes + settings.snapshot_reserve_bytes
    if free < required:
        raise RuntimeError(f"Insufficient snapshot storage ({free} free; {required} required)")
    try:
        with tarfile.open(partial, "w:gz") as tar:
            for item, arcname in entries:
                tar.add(item, arcname=arcname, recursive=False)
        with tarfile.open(partial, "r:gz") as tar:
            members = tar.getmembers()
            if not members or members[0].name.rstrip("/") != path.name:
                raise RuntimeError("Snapshot verification failed")
        os.replace(partial, archive)
        archive_bytes = archive.stat().st_size
        try:
            db.execute(
                "insert into snapshots(id,chat_id,path,kind,ref,created_at,source_bytes,archive_bytes,status) "
                "values(?,?,?,?,?,?,?,?,?)",
                (snap_id, chat_id, str(path), "tar", archive_name, db.utcnow(), source_bytes, archive_bytes, "ready"),
            )
        except Exception:
            archive.unlink(missing_ok=True)
            raise
    finally:
        partial.unlink(missing_ok=True)
    return {
        "id": snap_id,
        "path": str(path),
        "kind": "tar",
        "ref": archive_name,
        "source_bytes": source_bytes,
        "archive_bytes": archive_bytes,
        "status": "ready",
    }


def _archive_path(ref: str) -> Path:
    return settings.snapshot_dir / Path(ref).name


def discard_snapshot(snapshot_id: str) -> None:
    snap = db.one("select * from snapshots where id=?", (snapshot_id,))
    if not snap:
        return
    _archive_path(snap["ref"]).unlink(missing_ok=True)
    db.execute("delete from snapshots where id=?", (snapshot_id,))


def restore_snapshot(snapshot_id: str) -> dict[str, Any]:
    snap = db.one("select * from snapshots where id=?", (snapshot_id,))
    if not snap:
        raise HTTPException(404, "Snapshot not found")
    archive = _archive_path(snap["ref"])
    if not archive.exists():
        raise HTTPException(410, "Snapshot archive expired")
    target = allowed_path(snap["path"], must_exist=False)
    with tempfile.TemporaryDirectory(prefix="ollma-restore-", dir=str(target.parent)) as temp:
        temp_path = Path(temp)
        with tarfile.open(archive, "r:gz") as tar:
            for member in tar.getmembers():
                resolved = (temp_path / member.name).resolve()
                if temp_path.resolve() not in resolved.parents and resolved != temp_path.resolve():
                    raise RuntimeError("Unsafe snapshot archive")
            tar.extractall(temp_path, filter="data")
        restored = temp_path / target.name
        if not restored.exists():
            raise RuntimeError("Snapshot root missing")
        if target.is_file() or restored.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(restored, target)
        else:
            target.mkdir(parents=True, exist_ok=True)
            for child in list(target.iterdir()):
                if child.is_dir() and not child.is_symlink():
                    shutil.rmtree(child)
                else:
                    child.unlink(missing_ok=True)
            for child in restored.iterdir():
                destination = target / child.name
                if child.is_dir():
                    shutil.copytree(child, destination)
                else:
                    shutil.copy2(child, destination)
    return snap


def apply_stage(source: Path, stage: Path) -> dict[str, list[str]]:
    source = allowed_path(source)
    source_manifest = workspace_manifest(source)
    stage_manifest = workspace_manifest(stage)
    changed = sorted(k for k, value in stage_manifest.items() if source_manifest.get(k) != value)
    deleted = sorted(set(source_manifest) - set(stage_manifest))
    source_base = source.parent if source.is_file() else source
    for rel in changed:
        src = stage / rel
        dst = source_base / rel
        resolved = dst.resolve(strict=False)
        if source_base.resolve() not in resolved.parents and resolved != source_base.resolve():
            raise RuntimeError("Unsafe staged path")
        dst.parent.mkdir(parents=True, exist_ok=True)
        temp = dst.with_name(f".{dst.name}.ollma-tmp")
        shutil.copy2(src, temp)
        os.replace(temp, dst)
    for rel in deleted:
        target = source_base / rel
        if target.is_file() or target.is_symlink():
            target.unlink(missing_ok=True)
    return {"changed": changed, "deleted": deleted}


def cleanup_snapshots(days: int | None = None, dry_run: bool = False) -> int:
    cutoff = (datetime.now(UTC) - timedelta(days=days or settings.snapshot_retention_days)).isoformat()
    rows = db.all_rows(
        "select * from snapshots where created_at < ? and archive_deleted_at is null "
        "and id not in (select snapshot_id from runs where snapshot_id is not null and status not in ('completed','failed','cancelled','rolled_back'))",
        (cutoff,),
    )
    if dry_run:
        return len(rows)
    for row in rows:
        _archive_path(row["ref"]).unlink(missing_ok=True)
        db.execute("update snapshots set archive_deleted_at=? where id=?", (db.utcnow(), row["id"]))
    return len(rows)


def cleanup_partial_snapshots() -> int:
    cutoff = time.time() - settings.orphan_grace_hours * 3600
    removed = 0
    for item in settings.snapshot_dir.glob(".*.part"):
        with contextlib.suppress(OSError):
            if item.stat().st_mtime < cutoff:
                item.unlink()
                removed += 1
    return removed


def storage_report() -> dict[str, Any]:
    usage = shutil.disk_usage(settings.snapshot_dir)
    tracked_rows = db.all_rows("select id,ref,archive_bytes,archive_deleted_at from snapshots")
    tracked_refs = {Path(row["ref"]).name for row in tracked_rows}
    tracked_bytes = 0
    missing = []
    for row in tracked_rows:
        archive = _archive_path(row["ref"])
        if row.get("archive_deleted_at"):
            continue
        if archive.is_file():
            tracked_bytes += archive.stat().st_size
        else:
            missing.append(row["id"])
    orphans = []
    for archive in sorted(settings.snapshot_dir.glob("snap-*.tar.gz")):
        if archive.name in tracked_refs:
            continue
        stat = archive.stat()
        orphans.append(
            {
                "ref": archive.name,
                "bytes": stat.st_size,
                "modified_at": datetime.fromtimestamp(stat.st_mtime, UTC).isoformat(),
                "cleanup_eligible": stat.st_mtime < time.time() - settings.orphan_grace_hours * 3600,
            }
        )
    partials = [item.name for item in settings.snapshot_dir.glob(".*.part")]
    return {
        "filesystem": {"total_bytes": usage.total, "used_bytes": usage.used, "free_bytes": usage.free},
        "tracked": {"count": len(tracked_rows), "bytes": tracked_bytes, "missing_ids": missing},
        "orphans": orphans,
        "orphan_bytes": sum(item["bytes"] for item in orphans),
        "partials": partials,
        "limits": {
            "snapshot_max_bytes": settings.snapshot_max_bytes,
            "snapshot_reserve_bytes": settings.snapshot_reserve_bytes,
            "orphan_grace_hours": settings.orphan_grace_hours,
        },
    }


def cleanup_orphan_snapshots(refs: list[str], dry_run: bool = True) -> list[str]:
    report = storage_report()
    eligible = {item["ref"] for item in report["orphans"] if item["cleanup_eligible"]}
    selected = []
    for ref in refs:
        name = Path(ref).name
        if name != ref or name not in eligible:
            continue
        selected.append(name)
        if not dry_run:
            (settings.snapshot_dir / name).unlink(missing_ok=True)
    return selected
