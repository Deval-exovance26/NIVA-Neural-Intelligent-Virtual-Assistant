"""Environment verification for the NIVA offline pipeline.

Run on the EC2 g6e.4xlarge (L40S) to assert the GPU + CUDA torch stack is live:

    python -m niva_offline.verify_env

Invoked by `make verify-env` from the repo root. Exits non-zero if the GPU is not
visible or torch cannot see CUDA, so it doubles as a smoke test (docs/plan.md Task 1).
"""

from __future__ import annotations

import shutil
import subprocess
import sys


def _print_nvidia_smi() -> None:
    """Print driver / CUDA from nvidia-smi if available (non-fatal if missing)."""
    smi = shutil.which("nvidia-smi")
    if not smi:
        print("[warn] nvidia-smi not found on PATH (ok inside some containers).")
        return
    try:
        out = subprocess.run(
            [smi, "--query-gpu=name,driver_version,memory.total",
             "--format=csv,noheader"],
            capture_output=True, text=True, check=True,
        )
        print("[nvidia-smi] GPU(s):")
        for line in out.stdout.strip().splitlines():
            print(f"    {line.strip()}")
    except subprocess.CalledProcessError as exc:  # pragma: no cover - env dependent
        print(f"[warn] nvidia-smi failed: {exc}")


def main() -> int:
    print("=== NIVA offline-pipeline environment check ===")
    print(f"python: {sys.version.split()[0]}")

    _print_nvidia_smi()

    try:
        import torch
    except ImportError:
        print("[FAIL] torch is not installed. Run `make setup` first.")
        return 1

    print(f"torch: {torch.__version__}")
    print(f"torch CUDA build: {torch.version.cuda}")

    if not torch.cuda.is_available():
        print("[FAIL] torch.cuda.is_available() is False — no GPU visible to torch.")
        return 1

    device_count = torch.cuda.device_count()
    print(f"[OK] torch.cuda.is_available() == True ({device_count} device(s))")
    for i in range(device_count):
        name = torch.cuda.get_device_name(i)
        cap = torch.cuda.get_device_capability(i)
        print(f"    cuda:{i} -> {name} (sm_{cap[0]}{cap[1]})")

    # L40S is Ada / sm_89. Warn (don't fail) if we don't see it, so the check still
    # works on other dev GPUs.
    names = [torch.cuda.get_device_name(i) for i in range(device_count)]
    if not any("L40S" in n for n in names):
        print("[warn] No L40S detected — expected on the g6e.4xlarge target.")

    print("=== environment OK ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
