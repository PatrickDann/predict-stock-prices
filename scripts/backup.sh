#!/usr/bin/env bash
# Nightly Postgres backup -> off-box object storage (see docs/VISION_AND_ROADMAP.md §3.1).
#
# Self-hosting means YOU own disaster recovery. Run this from cron on the host:
#   30 3 * * *  /path/to/repo/scripts/backup.sh >> /var/log/mi-backup.log 2>&1
#
# Requires: pg_dump (or docker), and rclone configured with a remote (Cloudflare
# R2 / Backblaze B2 both have free tiers). Configure via env / .env.
set -euo pipefail

# --- config (override via environment) ---
DB_NAME="${DB_NAME:-market_intel}"
DB_USER="${DB_USER:-market_intel}"
DB_HOST="${DB_HOST:-localhost}"
DB_PORT="${DB_PORT:-5432}"
BACKUP_DIR="${BACKUP_DIR:-/tmp/mi-backups}"
RCLONE_REMOTE="${RCLONE_REMOTE:-}"        # e.g. "r2:market-intel-backups" ; empty = local only
RETENTION_DAYS="${RETENTION_DAYS:-14}"
# PGPASSWORD should be exported by the environment (do not hard-code secrets).

stamp="$(date +%Y%m%d-%H%M%S)"
mkdir -p "$BACKUP_DIR"
dump="$BACKUP_DIR/${DB_NAME}-${stamp}.dump"

echo "[$(date -Is)] pg_dump -> $dump"
pg_dump -Fc -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" "$DB_NAME" -f "$dump"

if [[ -n "$RCLONE_REMOTE" ]]; then
  echo "[$(date -Is)] uploading to $RCLONE_REMOTE"
  rclone copy "$dump" "$RCLONE_REMOTE"
fi

# Prune local dumps older than retention window.
find "$BACKUP_DIR" -name "${DB_NAME}-*.dump" -type f -mtime "+${RETENTION_DAYS}" -delete
echo "[$(date -Is)] done"
