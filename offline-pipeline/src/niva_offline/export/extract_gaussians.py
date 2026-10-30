"""Extract a browser-exportable Gaussian avatar from GAGAvatar (Stage 1).

Runs INSIDE the ``GAGAvatar`` conda env (needs torch 2.4.1, pytorch3d, the
32-channel ``diff_gaussian_rasterization_32d`` renderer, and the ``gagavatar``
package). It is launched as a standalone script from the GAGAvatar repo root so
its relative ``./assets`` paths resolve.

What it does
------------
1. Loads the GAGAvatar runtime + the tracked NIVA portrait (``tracked.pt``).
2. Builds the identity-fixed Gaussian parameters (``gs_params``) for a chosen
   near-frontal driving frame: positions (~58K), linear scales, unit-quaternion
   rotations, activated opacities. These are GAGAvatar's REAL geometry.
3. Renders that frame's final RGB (``sr_gen_image``, 512x512) — the true female
   avatar appearance — and projects every Gaussian center into that image using
   GAGAvatar's own camera matrices, sampling per-Gaussian RGB (option (a)).
   The raw 32-D learned feature "colors" are deliberately NOT used as RGB.
4. Writes the shared :mod:`niva_offline.export.gaussian_format` payload.

Color limitation (documented in the export meta): colors come from ONE near
frontal view, so surfaces occluded/back-facing in that view get approximate
colors. Task 8 refinement replaces RGB with video-optimized values.

Usage (from the GAGAvatar repo dir, inside the GAGAvatar env)::

    python extract_gaussians.py \
        --tracked render_results/tracked/tracked.pt \
        --avatar-id front.png \
        --out /opt/dlami/nvme/data/export/v1 \
        --niva-src <path to niva_offline src>
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch


def _quat_from_matrix_already(rotations: torch.Tensor) -> torch.Tensor:
    """GAGAvatar rotations are already unit quaternions (N,4). Pass-through,
    re-normalized for safety. Order is the rasterizer's (w, x, y, z)."""
    return torch.nn.functional.normalize(rotations, dim=-1)


def project_points_to_image(points: torch.Tensor, cam_matrix: torch.Tensor,
                            focal_x: float, focal_y: float, size) -> torch.Tensor:
    """Project world points (N,3) into pixel coords (N,2) using GAGAvatar's
    camera convention (mirrors utils_renderer.build_camera_matrices).

    Returns pixel (x, y) in [0, W)/[0, H) image space (float).
    """
    from gagavatar.libs.utils_renderer import build_camera_matrices

    view_mat, proj_mat, _ = build_camera_matrices(
        cam_matrix[None], focal_x, focal_y
    )
    view_mat = view_mat[0]          # (4,4)
    full_proj = proj_mat[0]         # (4,4) = view @ proj^T

    n = points.shape[0]
    homo = torch.cat([points, points.new_ones(n, 1)], dim=1)  # (N,4)
    clip = homo @ full_proj                                   # (N,4)
    w = clip[:, 3:4].clamp(min=1e-6)
    ndc = clip[:, :3] / w                                     # x,y in [-1,1]
    H, W = size
    # NDC (-1..1, +y up) -> pixel (0..W, 0..H, +y down)
    px = (ndc[:, 0] * 0.5 + 0.5) * W
    py = (1.0 - (ndc[:, 1] * 0.5 + 0.5)) * H
    return torch.stack([px, py], dim=1)


def sample_colors(image_rgb: torch.Tensor, pix: torch.Tensor) -> torch.Tensor:
    """Sample RGB (3,) per point from a (3,H,W) image at pixel coords (N,2).

    Uses nearest-pixel sampling with clamping. Returns (N,3) in [0,1].
    Points landing outside the image are clamped to the border (documented
    limitation: off-view Gaussians get edge colors as a fallback).
    """
    _, H, W = image_rgb.shape
    xs = pix[:, 0].round().long().clamp(0, W - 1)
    ys = pix[:, 1].round().long().clamp(0, H - 1)
    cols = image_rgb[:, ys, xs].transpose(0, 1)  # (N,3)
    return cols.clamp(0, 1)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Extract GAGAvatar Gaussians -> NIVA export")
    parser.add_argument("--tracked", required=True, type=Path)
    parser.add_argument("--avatar-id", default="front.png", type=str)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--resume-path", default="./assets/GAGAvatar.pt", type=str)
    parser.add_argument("--niva-src", required=True, type=Path,
                        help="path to offline-pipeline/src so niva_offline imports")
    parser.add_argument("--name", default="avatar", type=str)
    args = parser.parse_args(argv)

    sys.path.insert(0, str(args.niva_src.resolve()))
    from niva_offline.export.gaussian_format import GaussianAvatar, save

    device = "cuda"
    from gagavatar.runtime import GAGAvatarRuntime, GAGAvatarRuntimeConfig
    from gagavatar.assets import GAGAvatarAssets

    # --- load runtime + tracked avatar ------------------------------------
    assets = GAGAvatarAssets(
        root=Path("./assets"),
        model_path=Path(args.resume_path),
        tracked_path=args.tracked,
        flame_model_path=Path("./assets/FLAME_with_eye.pt"),
    )
    runtime = GAGAvatarRuntime(GAGAvatarRuntimeConfig(assets=assets, device=device))

    print("[extract] tracked avatars:", runtime.available_avatar_ids)
    runtime.set_avatar_id(args.avatar_id)

    # --- build the 106-D motion_code for this avatar's own (near-frontal) frame
    # motion_code layout (see runtime.build_forward_batch):
    #   [:100] expression, [100:103] head rotation, [103:106] jaw pose
    tracked = torch.load(args.tracked, map_location=device, weights_only=False)
    entry = tracked[args.avatar_id]
    exp = torch.as_tensor(entry["expcode"], device=device).float().reshape(-1)[:100]
    pose = torch.as_tensor(entry["posecode"], device=device).float().reshape(-1)  # (6,) head(3)+jaw(3)
    motion_code = torch.cat([exp, pose[:3], pose[3:6]], dim=0)[None]  # (1,106)

    batch = runtime.build_forward_batch(motion_code)
    gs = runtime.forward_gaussians_for_batch(batch)
    # take batch element 0
    xyz = gs["xyz"][0].detach()                      # (N,3)
    scales = gs["scales"][0].detach()                # (N,3) linear
    rotations = _quat_from_matrix_already(gs["rotations"][0].detach())  # (N,4)
    opacities = gs["opacities"][0].detach().clamp(0, 1)  # (N,1)
    n = xyz.shape[0]
    print(f"[extract] Gaussian count: {n}")

    # --- render true RGB for this frame + project to sample colors --------
    cam = runtime.model.cam_params
    img = runtime.render_rgb_batch(batch)[0].detach()   # (3,512,512) in [0,1]
    # project using the ACTUAL camera transform the renderer used
    t_transform = batch["t_transform"][0].detach()
    pix = project_points_to_image(
        xyz, t_transform,
        float(cam["focal_x"]), float(cam["focal_y"]), tuple(cam["size"]),
    )
    colors = sample_colors(img, pix)                    # (N,3)

    # fraction of gaussians that projected inside the image (color quality signal)
    H, W = cam["size"]
    inside = ((pix[:, 0] >= 0) & (pix[:, 0] < W) & (pix[:, 1] >= 0) & (pix[:, 1] < H))
    frac_inside = float(inside.float().mean())
    print(f"[extract] gaussians projecting inside frontal image: {frac_inside*100:.1f}%")

    avatar = GaussianAvatar(
        positions=xyz.cpu().numpy(),
        rotations=rotations.cpu().numpy(),
        scales=scales.cpu().numpy(),
        opacities=opacities.cpu().numpy(),
        colors=colors.cpu().numpy(),
        meta={
            "source": "GAGAvatar feed-forward base (Stage 1)",
            "subject": "niva-female",
            "avatar_id": args.avatar_id,
            "color_method": "sampled from near-frontal sr_gen_image (option a)",
            "frac_colored_from_view": round(frac_inside, 4),
            "limitations": (
                "RGB sampled from a single near-frontal render; occluded/back "
                "surfaces get border/approx colors. Refined in Task 8."
            ),
        },
    )
    paths = save(avatar, args.out, name=args.name)
    print(f"[OK] exported {n} gaussians -> {paths['bin']}")
    print(f"[OK] meta -> {paths['meta']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
