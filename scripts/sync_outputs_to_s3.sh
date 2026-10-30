#!/usr/bin/env bash
#
# sync_outputs_to_s3.sh — END OF DAY: push outputs NVMe/EBS -> S3 (before stop)
# ---------------------------------------------------------------------------
# Companion to sync_from_s3.sh. Run BEFORE stopping the instance so S3 holds
# the authoritative copy of every pipeline artifact. NVMe is wiped on stop; S3
# versioning keeps history. Idempotent (aws s3 sync only uploads changes).
#
# Maps local stage output dirs -> S3 stage prefixes. By default it looks for
# outputs under the scratch dir; override NIVA_OUTPUTS to point elsewhere.
#
# Usage:
#   bash scripts/sync_outputs_to_s3.sh                  # dry-run (shows plan)
#   NIVA_CONFIRM=1 bash scripts/sync_outputs_to_s3.sh   # actually upload
#   NIVA_CONFIRM=1 bash scripts/sync_outputs_to_s3.sh tracking refine
# ---------------------------------------------------------------------------
set -euo pipefail

log()  { printf '\033[1;36m[->s3]\033[0m %s\n' "$*"; }
ok()   { printf '\033[1;32m[ ok ]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[FAIL]\033[0m %s\n' "$*" >&2; exit 1; }

command -v aws >/dev/null 2>&1 || die "AWS CLI not found."

REGION="${NIVA_S3_REGION:-eu-central-1}"
SCRATCH="${NIVA_SCRATCH:-/mnt/nvme}"
OUTPUTS="${NIVA_OUTPUTS:-${SCRATCH}/data}"
CONFIRM="${NIVA_CONFIRM:-0}"

if [[ -n "${NIVA_S3_BUCKET:-}" ]]; then
  BUCKET="${NIVA_S3_BUCKET}"
else
  ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text 2>/dev/null || true)"
  [[ -n "${ACCOUNT_ID}" && "${ACCOUNT_ID}" != "None" ]] \
    || die "could not resolve AWS account id. Set NIVA_S3_BUCKET explicitly."
  BUCKET="niva-avatar-assets-${ACCOUNT_ID}-euc1"
fi

# Stage prefixes to push. raw-capture is excluded (immutable source, already
# in S3 from day 1) unless explicitly requested.
STAGES=("$@")
if [[ ${#STAGES[@]} -eq 0 ]]; then
  STAGES=(tracking base refine export)
fi

log "bucket  : s3://${BUCKET}"
log "region  : ${REGION}"
log "outputs : ${OUTPUTS}"
log "stages  : ${STAGES[*]}"

[[ -d "${OUTPUTS}" ]] || die "outputs dir ${OUTPUTS} not found."

DRY_FLAG="--dryrun"
[[ "${CONFIRM}" == "1" ]] && DRY_FLAG=""

for s in "${STAGES[@]}"; do
  src="${OUTPUTS}/${s}"
  if [[ -d "${src}" ]]; then
    log "sync ${src}/ -> s3://${BUCKET}/${s}/ ${DRY_FLAG:+(dry-run)}"
    aws s3 sync "${src}/" "s3://${BUCKET}/${s}/" --region "${REGION}" ${DRY_FLAG}
    ok "${s} ${CONFIRM:+done}"
  else
    warn "no local ${src}/ — skipping (stage not produced yet)."
  fi
done

if [[ "${CONFIRM}" != "1" ]]; then
  warn "DRY RUN. Re-run with NIVA_CONFIRM=1 to actually upload before stopping."
else
  ok "end-of-day sync complete. Safe to stop the instance now."
fi
