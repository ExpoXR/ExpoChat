#!/usr/bin/env bash
# WAL-safe SQLite backup with rotation and optional off-host copy.
#
# Env overrides:
#   DB_PATH        source database        (default: data/ollma.sqlite3)
#   BACKUP_DIR     local backup directory (default: backups)
#   BACKUP_KEEP    how many to retain     (default: 14)
#   BACKUP_OFFSITE optional dir/mount to also copy the newest archive into
#
# Schedule from cron/systemd, e.g. daily at 03:15:
#   15 3 * * * cd /path/to/ExpoChat && BACKUP_KEEP=30 scripts/backup.sh >> backups/backup.log 2>&1
set -euo pipefail

DB_PATH="${DB_PATH:-data/ollma.sqlite3}"
BACKUP_DIR="${BACKUP_DIR:-backups}"
BACKUP_KEEP="${BACKUP_KEEP:-14}"
BACKUP_OFFSITE="${BACKUP_OFFSITE:-}"

if [ ! -f "$DB_PATH" ]; then
  echo "backup: source database not found: $DB_PATH" >&2
  exit 1
fi

mkdir -p "$BACKUP_DIR"
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
archive="$BACKUP_DIR/ollma-$stamp.sqlite3"

# .backup is an online, WAL-consistent copy — safe while the app is running.
sqlite3 "$DB_PATH" ".backup '$archive'"
echo "backup: wrote $archive"

# Rotation: keep the newest BACKUP_KEEP archives, delete the rest.
mapfile -t archives < <(ls -1t "$BACKUP_DIR"/ollma-*.sqlite3 2>/dev/null || true)
if [ "${#archives[@]}" -gt "$BACKUP_KEEP" ]; then
  for stale in "${archives[@]:$BACKUP_KEEP}"; do
    rm -f -- "$stale"
    echo "backup: pruned $stale"
  done
fi

# Optional off-host copy (a mounted NAS share, USB target, etc.).
if [ -n "$BACKUP_OFFSITE" ]; then
  mkdir -p "$BACKUP_OFFSITE"
  cp -f -- "$archive" "$BACKUP_OFFSITE/"
  echo "backup: copied to $BACKUP_OFFSITE/"
fi
