#!/usr/bin/env bash
#
# setup_nvme_scratch.sh — format + mount the ephemeral instance-store NVMe
# ---------------------------------------------------------------------------
# The g6e.4xlarge ships a large local NVMe (instance-store, ~600GB). It is
# FAST and FREE but EPHEMERAL — wiped on every instance stop. We use it as
# scratch for bulky, regenerable data (decoded video frames, temp tensors,
# render dumps) and as the working copy of data synced from S3, keeping the
# small persistent EBS root lean.
#
# RUN THIS ON THE EC2 g6e.4xlarge AFTER EACH START (NVMe is blank on boot).
# Durable work must still land on EBS/S3 — never leave the only copy on NVMe
# before a stop (see RUNBOOK → stop/start).
#
# Usage:
#   sudo bash scripts/setup_nvme_scratch.sh            # format if blank + mount
#   NIVA_SCRATCH=/mnt/nvme sudo bash scripts/setup_nvme_scratch.sh
#   NIVA_NVME_FORCE_FORMAT=1 sudo bash scripts/setup_nvme_scratch.sh  # re-format
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

log()  { printf '\033[1;36m[nvme]\033[0m %s\n' "$*"; }
ok()   { printf '\033[1;32m[ ok ]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[FAIL]\033[0m %s\n' "$*" >&2; exit 1; }

MOUNT="${NIVA_SCRATCH:-/mnt/nvme}"
FORCE="${NIVA_NVME_FORCE_FORMAT:-0}"

[[ "$(id -u)" -eq 0 ]] || die "run with sudo (needs mkfs/mount)."

# --- Identify the instance-store NVMe --------------------------------------
# Instance-store devices report model 'Amazon EC2 NVMe Instance Storage'.
# The EBS root is 'Amazon Elastic Block Store' — we must NOT touch it.
log "scanning NVMe devices (will skip EBS root)..."
INSTANCE_STORE_DEV=""
for dev in /dev/nvme*n1; do
  [[ -b "${dev}" ]] || continue
  model="$(nvme id-ctrl "${dev}" 2>/dev/null | awk -F: '/^mn /{print $2}' | xargs || true)"
  if [[ "${model}" == *"Instance Storage"* ]]; then
    INSTANCE_STORE_DEV="${dev}"
    break
  fi
done

if [[ -z "${INSTANCE_STORE_DEV}" ]]; then
  # Fallback: lsblk heuristic — a disk with no partitions, not the mounted root.
  warn "nvme id-ctrl found no 'Instance Storage' device; trying lsblk fallback."
  root_src="$(findmnt -no SOURCE / | sed 's/[0-9]*$//')"
  INSTANCE_STORE_DEV="$(lsblk -dpno NAME,TYPE | awk '$2=="disk"{print $1}' \
    | grep -v "^${root_src}$" | grep nvme | head -n1 || true)"
fi

[[ -n "${INSTANCE_STORE_DEV}" ]] || die "no instance-store NVMe found. Does this instance type have one?"
log "instance-store device: ${INSTANCE_STORE_DEV}"

# --- Format if blank (or forced) -------------------------------------------
FSTYPE="$(blkid -o value -s TYPE "${INSTANCE_STORE_DEV}" 2>/dev/null || true)"
if [[ -z "${FSTYPE}" || "${FORCE}" == "1" ]]; then
  log "formatting ${INSTANCE_STORE_DEV} as ext4 (blank or NIVA_NVME_FORCE_FORMAT=1)..."
  mkfs.ext4 -F -L niva-scratch "${INSTANCE_STORE_DEV}"
  ok "formatted."
else
  ok "existing ${FSTYPE} filesystem found; not reformatting (set NIVA_NVME_FORCE_FORMAT=1 to wipe)."
fi

# --- Mount -----------------------------------------------------------------
mkdir -p "${MOUNT}"
if mountpoint -q "${MOUNT}"; then
  ok "${MOUNT} already mounted."
else
  mount "${INSTANCE_STORE_DEV}" "${MOUNT}"
  ok "mounted ${INSTANCE_STORE_DEV} -> ${MOUNT}"
fi

# Make it writable by the ubuntu user and lay out scratch subdirs.
chown -R ubuntu:ubuntu "${MOUNT}"
sudo -u ubuntu mkdir -p "${MOUNT}/data" "${MOUNT}/frames" "${MOUNT}/tmp" "${MOUNT}/renders"
ok "scratch layout ready under ${MOUNT}/ (data, frames, tmp, renders)."

cat <<EOF

Scratch is mounted at: ${MOUNT}
  ${MOUNT}/data     <- S3 working copy (fetched at start of day)
  ${MOUNT}/frames   <- decoded video frames (regenerable)
  ${MOUNT}/tmp      <- TMPDIR for runs
  ${MOUNT}/renders  <- intermediate render dumps

Point the pipeline at it, e.g.:
  export TMPDIR=${MOUNT}/tmp
  export NIVA_SCRATCH=${MOUNT}

REMINDER: this is EPHEMERAL. Anything you want to keep must be synced to
EBS/S3 before stopping the instance (scripts/sync_outputs_to_s3.sh).
EOF
