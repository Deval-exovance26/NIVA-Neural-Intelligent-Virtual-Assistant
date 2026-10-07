#!/usr/bin/env bash
#
# fetch_models.sh — NIVA 3DGS avatar dependency fetcher
# ---------------------------------------------------------------------------
# Idempotent downloader for every model / repo the offline pipeline needs.
# Everything lands under offline-pipeline/checkpoints/ (git-ignored).
#
# RUN THIS TOMORROW ON THE EC2 g6e.4xlarge (L40S, CUDA). It is authored to be
# run there; it does NOT run during local prep.
#
# Design rules:
#   * Idempotent: re-running skips anything already present.
#   * No license bypass: FLAME 2020 is license-gated behind a manual signup at
#     https://flame.is.tue.mpg.de/ — this script REFUSES to proceed without a
#     user-supplied asset path/credential; it never scrapes or circumvents.
#   * Each section prints exactly what it fetched.
#   * Exact upstream repo URLs are used (see KEY LOCKED DECISIONS in docs/plan.md).
#
# Usage:
#   bash scripts/fetch_models.sh                 # fetch everything it can
#   bash scripts/fetch_models.sh flame gagavatar # fetch only named sections
#   FLAME2020_SRC=/path/to/FLAME2020.zip bash scripts/fetch_models.sh flame
#
# Sections: flame | gagavatar | gagavatar_track | artalk | dinov3 | gsplat
# ---------------------------------------------------------------------------
set -euo pipefail

# --- Paths -----------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CKPT_DIR="${REPO_ROOT}/offline-pipeline/checkpoints"
REPOS_DIR="${CKPT_DIR}/repos"      # third-party git checkouts
WEIGHTS_DIR="${CKPT_DIR}/weights"  # downloaded model weights
mkdir -p "${REPOS_DIR}" "${WEIGHTS_DIR}"

# --- Helpers ---------------------------------------------------------------
log()  { printf '\033[1;36m[fetch]\033[0m %s\n' "$*"; }
ok()   { printf '\033[1;32m[ ok ]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[FAIL]\033[0m %s\n' "$*" >&2; exit 1; }

require_cmd() { command -v "$1" >/dev/null 2>&1 || die "required command '$1' not found on PATH"; }

# Clone-or-pull a git repo idempotently. $1=url $2=dest-dir-name [$3=ref]
clone_or_update() {
  local url="$1" name="$2" ref="${3:-}"
  local dest="${REPOS_DIR}/${name}"
  if [[ -d "${dest}/.git" ]]; then
    log "updating existing clone: ${name}"
    git -C "${dest}" fetch --depth 1 origin >/dev/null 2>&1 || warn "fetch failed for ${name} (offline?)"
  else
    log "cloning ${url} -> ${dest}"
    git clone --depth 1 "${url}" "${dest}"
  fi
  if [[ -n "${ref}" ]]; then
    git -C "${dest}" checkout "${ref}" >/dev/null 2>&1 || warn "could not checkout ref ${ref} for ${name}"
  fi
  ok "repo ready: ${dest}"
}

# ---------------------------------------------------------------------------
# SECTION: FLAME 2020  (LICENSE-GATED — manual signup required)
# ---------------------------------------------------------------------------
# FLAME 2020 is distributed only under the MPI license after registration at
# https://flame.is.tue.mpg.de/. We MUST NOT bypass that gate. The user must:
#   1. Register and accept the license at flame.is.tue.mpg.de
#   2. Download FLAME2020.zip (contains generic_model.pkl etc.)
#   3. Point FLAME2020_SRC at the downloaded zip OR the extracted folder.
# The teeth patch (patch_teeth.py -> 5143 verts / 10144 faces) and flame_uv.npz
# (256x256 UV, ~58,173 valid pixels) come bundled with the GAGAvatar repos.
fetch_flame() {
  log "=== FLAME 2020 (license-gated) ==="
  local flame_dir="${WEIGHTS_DIR}/flame2020"
  mkdir -p "${flame_dir}"

  if [[ -f "${flame_dir}/generic_model.pkl" ]]; then
    ok "FLAME 2020 already present at ${flame_dir}/generic_model.pkl — skipping"
    return 0
  fi

  local src="${FLAME2020_SRC:-}"
  if [[ -z "${src}" ]]; then
    die "FLAME 2020 is license-gated. Register + download at https://flame.is.tue.mpg.de/
      then re-run with:
        FLAME2020_SRC=/path/to/FLAME2020.zip bash scripts/fetch_models.sh flame
      (or point FLAME2020_SRC at an already-extracted folder containing generic_model.pkl)
      This script will NOT attempt to bypass the FLAME license."
  fi

  if [[ -f "${src}" && "${src}" == *.zip ]]; then
    require_cmd unzip
    log "extracting user-provided FLAME zip: ${src}"
    unzip -o "${src}" -d "${flame_dir}" >/dev/null
  elif [[ -d "${src}" ]]; then
    log "copying user-provided FLAME folder: ${src}"
    cp -r "${src}/." "${flame_dir}/"
  else
    die "FLAME2020_SRC='${src}' is neither a .zip file nor a directory."
  fi

  [[ -f "${flame_dir}/generic_model.pkl" ]] \
    || die "generic_model.pkl not found after import — check your FLAME download."
  ok "FLAME 2020 imported -> ${flame_dir}/generic_model.pkl"
}

# ---------------------------------------------------------------------------
# SECTION: GAGAvatar  (feed-forward base, Stage 1 stand-in)
# ---------------------------------------------------------------------------
# Repo: https://github.com/xg-chu/GAGAvatar
# Pretrained weights are published by the repo (HuggingFace / Google Drive as
# documented in its README). We clone the repo and invoke its own documented
# weight-download path so we never second-guess their hosting.
fetch_gagavatar() {
  log "=== GAGAvatar (feed-forward base) ==="
  require_cmd git
  clone_or_update "https://github.com/xg-chu/GAGAvatar.git" "GAGAvatar"
  local repo="${REPOS_DIR}/GAGAvatar"
  log "GAGAvatar weight acquisition (per repo README):"
  log "  The repo documents fetching assets via its helper (typically):"
  log "    cd ${repo} && python build_resources.py   # or ./build_resources.sh"
  log "  which pulls FLAME-derived resources + pretrained GAGAvatar weights."
  if [[ -f "${repo}/build_resources.py" ]]; then
    warn "Found build_resources.py. Review it, then run it manually to pull weights:"
    warn "    (cd ${repo} && python build_resources.py)"
  else
    warn "Could not auto-detect the weight helper; follow ${repo}/README.md."
  fi
  ok "GAGAvatar repo staged -> ${repo}"
}

# ---------------------------------------------------------------------------
# SECTION: GAGAvatar_track  (FLAME tracker / preprocessor)
# ---------------------------------------------------------------------------
# Repo: https://github.com/xg-chu/GAGAvatar_track
# EMICA-based tracker -> per-frame FLAME params + camera + bbox + landmarks.
# Ships flame_uv.npz and the teeth-patched FLAME topology helpers.
fetch_gagavatar_track() {
  log "=== GAGAvatar_track (FLAME tracker) ==="
  require_cmd git
  clone_or_update "https://github.com/xg-chu/GAGAvatar_track.git" "GAGAvatar_track"
  local repo="${REPOS_DIR}/GAGAvatar_track"
  log "GAGAvatar_track resource acquisition (per repo README):"
  log "  Typically: cd ${repo} && ./build_resources.sh"
  log "  Downloads tracker weights (EMICA/MICA/mediapipe assets) + flame_uv.npz."
  if [[ -f "${repo}/build_resources.sh" ]]; then
    warn "Found build_resources.sh. Review it, then run manually:"
    warn "    (cd ${repo} && bash build_resources.sh)"
  else
    warn "Could not auto-detect the resource helper; follow ${repo}/README.md."
  fi
  ok "GAGAvatar_track repo staged -> ${repo}"
}

# ---------------------------------------------------------------------------
# SECTION: ARTalk  (audio -> FLAME motioncode @ 25fps)
# ---------------------------------------------------------------------------
# Repo: https://github.com/xg-chu/ARTalk   (successor: xg-chu/UniLS)
fetch_artalk() {
  log "=== ARTalk (audio -> FLAME motion) ==="
  require_cmd git
  clone_or_update "https://github.com/xg-chu/ARTalk.git" "ARTalk"
  local repo="${REPOS_DIR}/ARTalk"
  log "ARTalk weights are published by the repo (see its README for the"
  log "  pretrained motion-generator + FLAME motioncode checkpoints)."
  warn "Follow ${repo}/README.md to pull ARTalk checkpoints (HF/Drive)."
  log "Successor model to evaluate later (Task 12): https://github.com/xg-chu/UniLS"
  ok "ARTalk repo staged -> ${repo}"
}

# ---------------------------------------------------------------------------
# SECTION: DINOv3  (frozen ViT-B/16 feature backbone)
# ---------------------------------------------------------------------------
# Repo: https://github.com/facebookresearch/dinov3
# Only required if a from-scratch SpatialAvatar-0 generator is built; cloned
# here so the backbone + loading code are available offline.
fetch_dinov3() {
  log "=== DINOv3 (facebookresearch/dinov3) ==="
  require_cmd git
  clone_or_update "https://github.com/facebookresearch/dinov3.git" "dinov3"
  local repo="${REPOS_DIR}/dinov3"
  log "DINOv3 ViT-B/16 weights are gated via Meta's request form / HuggingFace"
  log "  (facebook/dinov3-vitb16-*). Accept terms and download per ${repo}/README.md."
  warn "ViT-B/16 weights require accepting Meta's DINOv3 license — do that manually."
  ok "DINOv3 repo staged -> ${repo}"
}

# ---------------------------------------------------------------------------
# SECTION: gsplat  (Gaussian rasterization backend)
# ---------------------------------------------------------------------------
# Repo: https://github.com/nerfstudio-project/gsplat  (Apache-2.0)
# Preferred install is pip (compiles CUDA ext against the local toolchain).
fetch_gsplat() {
  log "=== gsplat (Gaussian rasterizer) ==="
  if python -c "import gsplat" >/dev/null 2>&1; then
    ok "gsplat already importable in the active environment — skipping"
    return 0
  fi
  log "Install into the active (CUDA) Python env with:"
  log "    pip install gsplat"
  log "  (Building the CUDA extension needs the L40S toolchain present — i.e. on EC2.)"
  log "  Source fallback: https://github.com/nerfstudio-project/gsplat"
  require_cmd git
  clone_or_update "https://github.com/nerfstudio-project/gsplat.git" "gsplat"
  warn "Run 'pip install gsplat' (or pip install -e ${REPOS_DIR}/gsplat) on the GPU box."
  ok "gsplat source staged -> ${REPOS_DIR}/gsplat"
}

# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
run_all() {
  fetch_flame
  fetch_gagavatar
  fetch_gagavatar_track
  fetch_artalk
  fetch_dinov3
  fetch_gsplat
}

main() {
  require_cmd git
  log "checkpoints dir: ${CKPT_DIR}"
  if [[ $# -eq 0 ]]; then
    run_all
  else
    for section in "$@"; do
      case "${section}" in
        flame)            fetch_flame ;;
        gagavatar)        fetch_gagavatar ;;
        gagavatar_track)  fetch_gagavatar_track ;;
        artalk)           fetch_artalk ;;
        dinov3)           fetch_dinov3 ;;
        gsplat)           fetch_gsplat ;;
        *) die "unknown section '${section}' (valid: flame gagavatar gagavatar_track artalk dinov3 gsplat)" ;;
      esac
    done
  fi
  ok "fetch_models.sh complete. Contents of ${CKPT_DIR}:"
  find "${CKPT_DIR}" -maxdepth 2 -mindepth 1 2>/dev/null | sort || true
}

main "$@"
