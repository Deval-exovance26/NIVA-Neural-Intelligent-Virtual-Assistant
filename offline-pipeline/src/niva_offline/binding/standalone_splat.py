"""S2 — Standalone SH Gaussian splat: init from FLAME triangles + plain render.

Builds a self-sufficient 3D Gaussian splat (NO neural renderer): one Gaussian per
FLAME triangle, rigged via the S1 binding core, with spherical-harmonics colour,
rendered through gsplat's standard SH rasterizer. This is the representation that
can look good as a plain browser splat.

Pipeline here (offline, on the box):
    FLAME mesh (posed per frame via GAGAvatar runtime)
      -> compute_triangle_frames (S1)
      -> bind one Gaussian per triangle (local offset 0, local rot I, local
         scale set so the splat ~ covers its triangle)
      -> SH colour (DC initialised from a sampled frontal image; rest = 0)
      -> gsplat.rasterization(..., sh_degree) -> RGB image (no StyleUNet)

Camera: GAGAvatar provides `cam_params` (focal_x, focal_y, size) + a per-frame
`t_transform` (world->camera-ish). We convert to gsplat's `viewmats` (world->cam,
OpenCV convention) + pinhole `K`. The exact convention is validated empirically by
the S2 probe (frontal render must be upright and recognisable).

Run in the GAGAvatar env (has torch 2.4.1 + gsplat + the FLAME runtime).
Requires CUDA_HOME=/usr/local/cuda-12.8 and the env bin on PATH for gsplat JIT.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from niva_offline.binding.triangle_binding import (
    bind_gaussians,
    compute_triangle_frames,
)

# SH degree-0 constant (3DGS convention): DC coeff relates to colour via C0.
C0 = 0.28209479177387814


def rgb_to_sh_dc(rgb: torch.Tensor) -> torch.Tensor:
    """Map linear RGB in [0,1] to the SH DC coefficient (band 0)."""
    return (rgb - 0.5) / C0


def sh_dc_to_rgb(dc: torch.Tensor) -> torch.Tensor:
    return (dc * C0 + 0.5).clamp(0, 1)


@dataclass
class StandaloneSplat:
    """A self-sufficient SH Gaussian splat with FLAME triangle bindings.

    LOCAL params (optimizable); world params are produced per frame by binding.
        local_offset (N,3), local_quat (N,4 wxyz), local_scale (N,3 linear),
        opacity_logit (N,), sh (N, (deg+1)^2, 3).
        parent_idx (N,) long -> FLAME triangle per Gaussian.
        faces (F,3) long -> FLAME topology (needed to rebuild frames each frame).
    """

    local_offset: torch.Tensor
    local_quat: torch.Tensor
    local_scale: torch.Tensor
    opacity_logit: torch.Tensor
    sh: torch.Tensor
    parent_idx: torch.Tensor
    faces: torch.Tensor
    sh_degree: int

    @property
    def num_gaussians(self) -> int:
        return int(self.parent_idx.shape[0])


def init_splat_from_faces(
    faces: torch.Tensor,
    verts_canonical: torch.Tensor,
    sh_degree: int = 3,
    init_rgb: torch.Tensor | None = None,
    scale_frac: float = 0.5,
    splats_per_face: int = 1,
    device: str = "cuda",
) -> StandaloneSplat:
    """Create ``splats_per_face`` SH Gaussians per FLAME triangle.

    More splats/face = more primitives = sharper detail (the fix for the
    "too few Gaussians -> blurry" ceiling). Each splat keeps its parent triangle
    binding (binding inheritance by construction). Extra splats are offset to
    distinct in-plane barycentric-ish positions within the triangle's local frame
    so they tile the surface instead of stacking.

    Args:
        faces: (F,3) long.
        verts_canonical: (V,3) a canonical/neutral posed mesh (for initial scale).
        sh_degree: SH band count = (deg+1)^2 coeffs.
        init_rgb: (F,3) optional per-triangle initial colour in [0,1]; else gray.
        scale_frac: initial Gaussian scale as a fraction of triangle size (k).
        splats_per_face: K>=1 Gaussians per triangle.
    """
    F = faces.shape[0]
    faces = faces.long().to(device)
    verts_canonical = verts_canonical.to(device)
    K = max(1, int(splats_per_face))
    N = F * K

    # In-plane offset pattern per splat (local frame x,y; z=0 stays on surface).
    # 1: centroid; >1: a small ring/grid around centroid, scaled < triangle.
    if K == 1:
        offs = torch.zeros(1, 3, device=device)
    else:
        ang = torch.linspace(0, 2 * math.pi, K, device=device)[:-0 or None]
        ring = torch.stack([torch.cos(ang), torch.sin(ang),
                            torch.zeros_like(ang)], dim=-1) * 0.33
        offs = torch.cat([torch.zeros(1, 3, device=device), ring[: K - 1]], dim=0)

    local_offset = offs.repeat(F, 1)  # (N,3) tiled per face

    local_quat = torch.zeros(N, 4, device=device)
    local_quat[:, 0] = 1.0  # identity (w,x,y,z)
    local_scale = torch.full((N, 3), scale_frac / max(1, K ** 0.5), device=device)
    local_scale[:, 2] *= 0.25  # flatten along triangle normal -> disc-like

    opacity_logit = torch.full((N,), _inverse_sigmoid(0.9), device=device)

    n_coeffs = (sh_degree + 1) ** 2
    sh = torch.zeros(N, n_coeffs, 3, device=device)
    if init_rgb is None:
        init_rgb = torch.full((N, 3), 0.6, device=device)
    elif init_rgb.shape[0] == F and K > 1:
        init_rgb = init_rgb.repeat_interleave(K, dim=0)
    sh[:, 0, :] = rgb_to_sh_dc(init_rgb.to(device))

    # parent triangle per splat: face f owns splats [f*K : (f+1)*K]
    parent_idx = torch.arange(F, device=device, dtype=torch.long).repeat_interleave(K)

    return StandaloneSplat(
        local_offset=local_offset, local_quat=local_quat, local_scale=local_scale,
        opacity_logit=opacity_logit, sh=sh, parent_idx=parent_idx, faces=faces,
        sh_degree=sh_degree,
    )


def world_gaussians(splat: StandaloneSplat, verts_posed: torch.Tensor):
    """Produce world-space (pos, quat, scale) for a posed FLAME mesh via binding.

    verts_posed: (V,3) the FLAME mesh for THIS frame. Returns (pos N3, quat N4,
    scale N3) ready for the rasterizer.
    """
    b, R, k = compute_triangle_frames(verts_posed, splat.faces)
    return bind_gaussians(
        splat.local_offset, splat.local_quat, splat.local_scale,
        splat.parent_idx, b, R, k,
    )


def _inverse_sigmoid(x: float) -> float:
    return math.log(x / (1.0 - x))
