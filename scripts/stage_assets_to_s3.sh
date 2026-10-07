#!/usr/bin/env bash
#
# stage_assets_to_s3.sh — upload NIVA capture assets to S3
# ---------------------------------------------------------------------------
# Uploads the large/binary capture assets that must NOT live in git:
#   * NIVA/video/NIVA_PSR.mp4   (reference video, ~534 MB)
#   * NIVA/images/*.png         (5 portraits, 1176x1337)
#   * NIVA/voice/*.mp3          (test audio)
# into an S3 bucket, preserving a clean prefix layout for the pipeline.
#
# RUN THIS TOMORROW (or whenever you are ready to populate S3). It is authored
# to be run manually and will REFUSE to proceed until you confirm the bucket.
#
# Bucket / region are env vars with documented DEFAULT SUGGESTIONS:
#   NIVA_S3_BUCKET   default: niva-avatar-assets-<accountid>   (resolved live)
#   NIVA_S3_REGION   default: us-east-2  (SOW region, US East / Ohio)
#   NIVA_S3_PREFIX   default: raw-capture/v1
#
# Usage:
#   # dry run (prints what WOULD upload, no writes):
#   NIVA_CONFIRM=0 bash scripts/stage_assets_to_s3.sh
#   # real upload (requires explicit confirmation):
#   NIVA_CONFIRM=1 bash scripts/stage_assets_to_s3.sh
#   # override bucket/region:
#   NIVA_S3_BUCKET=my-bucket NIVA_S3_REGION=us-east-2 NIVA_CONFIRM=1 \
#     bash scripts/stage_assets_to_s3.sh
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
NIVA_DIR="${REPO_ROOT}/NIVA"

log()  { printf '\033[1;36m[s3]\033[0m %s\n' "$*"; }
ok()   { printf '\033[1;32m[ ok ]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[FAIL]\033[0m %s\n' "$*" >&2; exit 1; }

command -v aws >/dev/null 2>&1 || die "AWS CLI not found. Install + 'aws configure' first."

REGION="${NIVA_S3_REGION:-us-east-2}"      # SOW region: US East (Ohio)
PREFIX="${NIVA_S3_PREFIX:-raw-capture/v1}"
CONFIRM="${NIVA_CONFIRM:-0}"

# Resolve the default bucket name from the caller's AWS account id, unless the
# user overrode NIVA_S3_BUCKET. We do NOT create anything without confirmation.
if [[ -n "${NIVA_S3_BUCKET:-}" ]]; then
  BUCKET="${NIVA_S3_BUCKET}"
else
  log "resolving AWS account id for default bucket name..."
  ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text 2>/dev/null || true)"
  [[ -n "${ACCOUNT_ID}" && "${ACCOUNT_ID}" != "None" ]] \
    || die "could not resolve AWS account id (is 'aws configure' done?). Set NIVA_S3_BUCKET explicitly."
  BUCKET="niva-avatar-assets-${ACCOUNT_ID}"
fi

log "target bucket : s3://${BUCKET}"
log "target region : ${REGION}"
log "target prefix : ${PREFIX}"
log "source dir    : ${NIVA_DIR}"

[[ -d "${NIVA_DIR}" ]] || die "NIVA/ not found at ${NIVA_DIR}"

# --- bucket guard: ls, else offer to mb (only with confirmation) -----------
if aws s3 ls "s3://${BUCKET}" --region "${REGION}" >/dev/null 2>&1; then
  ok "bucket s3://${BUCKET} exists and is accessible."
else
  warn "bucket s3://${BUCKET} does not exist / not accessible."
  if [[ "${CONFIRM}" == "1" ]]; then
    log "creating bucket s3://${BUCKET} in ${REGION}..."
    if [[ "${REGION}" == "us-east-1" ]]; then
      aws s3 mb "s3://${BUCKET}" --region "${REGION}"
    else
      aws s3api create-bucket --bucket "${BUCKET}" --region "${REGION}" \
        --create-bucket-configuration "LocationConstraint=${REGION}"
    fi
    ok "bucket created."
  else
    warn "NIVA_CONFIRM!=1 so NOT creating the bucket. Re-run with NIVA_CONFIRM=1 to create + upload."
  fi
fi

# --- enumerate assets ------------------------------------------------------
VIDEO_SRC="${NIVA_DIR}/video/NIVA_PSR.mp4"
declare -a UPLOADS=()  # "local|s3key" pairs

[[ -f "${VIDEO_SRC}" ]] && UPLOADS+=("${VIDEO_SRC}|${PREFIX}/video/NIVA_PSR.mp4") \
  || warn "reference video missing: ${VIDEO_SRC}"

shopt -s nullglob
for png in "${NIVA_DIR}"/images/*.png; do
  UPLOADS+=("${png}|${PREFIX}/images/$(basename "${png}")")
done
for mp3 in "${NIVA_DIR}"/voice/*.mp3; do
  UPLOADS+=("${mp3}|${PREFIX}/voice/$(basename "${mp3}")")
done
shopt -u nullglob

[[ ${#UPLOADS[@]} -gt 0 ]] || die "no assets found to upload under ${NIVA_DIR}"

log "planned uploads (${#UPLOADS[@]}):"
for pair in "${UPLOADS[@]}"; do
  local_path="${pair%%|*}"; s3key="${pair##*|}"
  size="$(du -h "${local_path}" 2>/dev/null | cut -f1)"
  printf '    %-8s %s  ->  s3://%s/%s\n' "[${size}]" "$(basename "${local_path}")" "${BUCKET}" "${s3key}"
done

# --- confirmation gate -----------------------------------------------------
if [[ "${CONFIRM}" != "1" ]]; then
  warn "DRY RUN. No uploads performed."
  warn "Confirm the bucket/region above, then re-run with NIVA_CONFIRM=1 to upload."
  exit 0
fi

# --- upload ----------------------------------------------------------------
for pair in "${UPLOADS[@]}"; do
  local_path="${pair%%|*}"; s3key="${pair##*|}"
  log "uploading $(basename "${local_path}") ..."
  aws s3 cp "${local_path}" "s3://${BUCKET}/${s3key}" --region "${REGION}"
  ok "uploaded -> s3://${BUCKET}/${s3key}"
done

ok "all assets staged. Listing s3://${BUCKET}/${PREFIX}/ :"
aws s3 ls "s3://${BUCKET}/${PREFIX}/" --recursive --region "${REGION}"
