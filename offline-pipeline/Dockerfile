# =============================================================================
# NIVA offline-pipeline — reproducible CUDA 12.x build image
# =============================================================================
# Target host: EC2 g6e.4xlarge — NVIDIA L40S (Ada Lovelace, sm_89, 48GB).
# Base: official NVIDIA CUDA 12.1 *devel* image (Ubuntu 22.04). The devel image
# ships nvcc + headers so CUDA extensions (gsplat, GAGAvatar ops) can be compiled
# on-instance against sm_89.
#
# Build tomorrow on the instance:
#   docker build -t niva-offline:latest offline-pipeline/
# Run (needs the NVIDIA container toolkit on the host):
#   docker run --rm --gpus all niva-offline:latest python -m niva_offline.verify_env
#
# NOTE: model weights and datasets are NOT baked into the image. Fetch them on the
# instance (scripts/fetch_models.sh) and mount data via a volume / pull from S3.
# =============================================================================

FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Non-interactive apt; make sm_89 the compile target for any CUDA extensions.
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TORCH_CUDA_ARCH_LIST="8.9" \
    FORCE_CUDA=1

# -- OS packages: python 3.10, build toolchain, git, OpenGL/ffmpeg for cv/video --
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 \
        python3.10-venv \
        python3.10-dev \
        python3-pip \
        git \
        build-essential \
        ninja-build \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Make python3.10 the default `python` / `pip`.
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && python -m pip install --upgrade "pip==24.2" "setuptools==68.2.2" "wheel==0.41.2"

WORKDIR /opt/niva/offline-pipeline

# -- Install the pinned Python stack first (better layer caching) --------------
# Copy only dependency manifests before the source so edits to code don't bust
# the (slow) torch install layer.
COPY requirements.txt ./
RUN pip install -r requirements.txt

# -- Install the package itself (src/ layout) ----------------------------------
COPY pyproject.toml README.md ./
COPY src ./src
COPY tests ./tests
RUN pip install -e .

# -----------------------------------------------------------------------------
# GPU / git-sourced deps that JIT-compile CUDA extensions against sm_89.
# Uncomment + pin commits once confirmed on-instance (kept out of the base build
# so this image stays buildable without network access to those repos):
#
#   RUN pip install "gsplat==1.3.0"
#   RUN pip install "git+https://github.com/xg-chu/GAGAvatar.git@<PIN_COMMIT>"
#   RUN pip install "git+https://github.com/xg-chu/GAGAvatar_track.git@<PIN_COMMIT>"
#   RUN pip install "git+https://github.com/xg-chu/ARTalk.git@<PIN_COMMIT>"
#   # GaussianAvatars = Toyota NON-COMMERCIAL — see LICENSES.md before vendoring.
# -----------------------------------------------------------------------------

# Default: print the environment check. Needs `--gpus all` at run time.
CMD ["python", "-m", "niva_offline.verify_env"]
