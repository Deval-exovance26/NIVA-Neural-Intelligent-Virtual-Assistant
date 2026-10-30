"""Offline validation renderer for the NIVA Gaussian export.

Pure-numpy point/splat rasterizer used to VISUALLY confirm the exported avatar
looks correct (identity preserved) before/independently of the WebGPU browser
path. Not for production — it approximates each Gaussian as a small additive
disc. It uses the SAME camera conventions (right-handed lookAt + perspective)
the browser renderer uses, so a correct image here means the export + camera
math are consistent.

Usage:
    python -m niva_offline.export.validate_render \
        --export-dir /opt/dlami/nvme/data/export/v1 --name niva_base \
        --out /tmp/niva_val_front.png --yaw 0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from niva_offline.export.gaussian_format import load


def look_at(eye, center, up):
    z = eye - center
    z = z / (np.linalg.norm(z) + 1e-9)
    x = np.cross(up, z)
    x = x / (np.linalg.norm(x) + 1e-9)
    y = np.cross(z, x)
    R = np.stack([x, y, z], axis=0)
    t = -R @ eye
    M = np.eye(4, dtype=np.float32)
    M[:3, :3] = R
    M[:3, 3] = t
    return M


def render(avatar, width=512, height=512, yaw=0.0, pitch=0.0, fovy=0.6):
    pos = avatar.positions.astype(np.float64)
    col = avatar.colors.astype(np.float64)
    op = avatar.opacities.reshape(-1).astype(np.float64)
    scl = avatar.scales.mean(1).astype(np.float64)

    center = pos.mean(0)
    radius = np.linalg.norm(pos - center, axis=1).max()
    dist = radius * 2.6
    eye = center + dist * np.array([
        np.cos(pitch) * np.sin(yaw), np.sin(pitch), np.cos(pitch) * np.cos(yaw)
    ])
    V = look_at(eye.astype(np.float32), center.astype(np.float32),
                np.array([0, -1, 0], np.float32))  # GAGAvatar world is +Y down

    homo = np.concatenate([pos, np.ones((pos.shape[0], 1))], axis=1)
    vcam = homo @ V.T                      # (N,4) view space
    zc = -vcam[:, 2]
    valid = zc > 1e-3
    f = 1.0 / np.tan(fovy / 2)
    aspect = width / height
    ndc_x = (f / aspect) * vcam[:, 0] / zc
    ndc_y = f * vcam[:, 1] / zc
    px = ((ndc_x * 0.5 + 0.5) * width).astype(int)
    py = ((1.0 - (ndc_y * 0.5 + 0.5)) * height).astype(int)

    # painter's order: far -> near
    order = np.argsort(-zc)
    img = np.full((height, width, 3), (0.06, 0.07, 0.09), dtype=np.float64)
    acc = np.zeros((height, width), dtype=np.float64)

    # pixel radius from world scale (perspective)
    rad = np.clip((scl * f / np.maximum(zc, 1e-3)) * (height * 0.5), 0.6, 6.0)

    for i in order:
        if not valid[i]:
            continue
        x, y = px[i], py[i]
        if x < 0 or x >= width or y < 0 or y >= height:
            continue
        a = op[i]
        if a < 0.02:
            continue
        rr = int(rad[i])
        x0, x1 = max(0, x - rr), min(width, x + rr + 1)
        y0, y1 = max(0, y - rr), min(height, y + rr + 1)
        if x0 >= x1 or y0 >= y1:
            img[y, x] = img[y, x] * (1 - a) + col[i] * a
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        d2 = ((xx - x) ** 2 + (yy - y) ** 2) / max(rr * rr, 1)
        w = np.exp(-2.5 * d2) * a
        for c in range(3):
            img[y0:y1, x0:x1, c] = img[y0:y1, x0:x1, c] * (1 - w) + col[i, c] * w

    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--export-dir", required=True, type=Path)
    p.add_argument("--name", default="avatar")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--yaw", type=float, default=0.0)
    p.add_argument("--pitch", type=float, default=0.0)
    args = p.parse_args(argv)

    avatar = load(args.export_dir, name=args.name)
    img = render(avatar, yaw=args.yaw, pitch=args.pitch)
    try:
        import imageio.v2 as imageio
        imageio.imwrite(args.out, img)
    except Exception:
        import cv2
        cv2.imwrite(str(args.out), img[:, :, ::-1])
    print(f"[OK] wrote {args.out} ({avatar.num_gaussians} gaussians)")
    # report simple content stats so correctness is checkable without a viewer
    nonbg = (img.reshape(-1, 3).std(1) > 5).mean()
    print(f"[stat] non-background pixel fraction: {nonbg*100:.1f}%  mean RGB: {img.reshape(-1,3).mean(0).round(1)}")


if __name__ == "__main__":
    raise SystemExit(main())
