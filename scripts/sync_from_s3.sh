#!/usr/bin/env bash
#
# sync_from_s3.sh — START OF DAY: pull data artifacts from S3 -> NVMe scratch
# ---------------------------------------------------------------------------
# Companion to sync_outputs_to_s3.sh. S3 is the source of truth; NVMe is the
# fast ephemeral working copy. Run this after setup_nvme_scratch.sh each
# morning to repopulate scratch so you can resume where you left off.
#
# The conda env + model weights + FLAME live on the PERSISTENT EBS root and do
# NOT need re-fetching — only data/artifacts come from S3.
#
# Usage:
#   bash scripts/sync_from_s3.sh                 # pull all stages present in S3
#   bash scripts/sync_from_s3.sh raw-capture tracking   # only named prefixes
#   NIVA_SCRATCH=/mnt/nvme bash scripts/sync_from_s3.sh
# ---------------------------------------------------------------------------
set -euo pipefail

log()  { printf '\033[1;36m[s3<-]\033[0m %s\n' "$*"; }
ok()   { printf '\033[1;32m[ ok ]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[FAIL]\033[0m %s\n' "$*" >&2; exit 1; }

command -v aws >/dev/null 2>&1 || die "AWS CLI not found."

REGION="${NIVA_S3_REGION:-eu-central-1}"
SCRATCH="${NIVA_SCRATCH:-/mnt/nvme}"
DEST="${SCRATCH}/data"

# Resolve bucket (same convention as stage_assets_to_s3.sh).
if [[ -n "${NIVA_S3_BUCKET:-}" ]]; then
  BUCKET="${NIVA_S3_BUCKET}"
else
  ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text 2>/dev/null || true)"
  [[ -n "${ACCOUNT_ID}" && "${ACCOUNT_ID}" != "None" ]] \
    || die "could not resolve AWS account id. Set NIVA_S3_BUCKET explicitly."
  BUCKET="niva-avatar-assets-${ACCOUNT_ID}-euc1"
fi

# Default: all known stage prefixes. Override by passing names as args.
PREFIXES=("$@")
if [[ ${#PREFIXES[@]} -eq 0 ]]; then
  PREFIXES=(raw-capture tracking base refine export)
fi

[[ -d "${SCRATCH}" ]] || die "scratch dir ${SCRATCH} missing — run setup_nvme_scratch.sh first."
mkdir -p "${DEST}"

log "bucket : s3://${BUCKET}"
log "region : ${REGION}"
log "dest   : ${DEST}"

for p in "${PREFIXES[@]}"; do
  if aws s3 ls "s3://${BUCKET}/${p}/" --region "${REGION}" >/dev/null 2>&1; then
    log "syncing ${p}/ ..."
    aws s3 sync "s3://${BUCKET}/${p}/" "${DEST}/${p}/" --region "${REGION}"
    ok "synced ${p}/"
  else
    warn "no s3://${BUCKET}/${p}/ yet — skipping (stage not produced)."
  fi
done

ok "start-of-day sync complete. Working data in ${DEST}/"
