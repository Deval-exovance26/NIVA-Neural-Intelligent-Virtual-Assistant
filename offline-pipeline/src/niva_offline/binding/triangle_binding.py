"""FLAME triangle binding core — GaussianAvatars Eq. 1 (clean reimplementation).

Each Gaussian is rigged to a *parent triangle* of the FLAME mesh. Per frame we
derive a local coordinate frame from the triangle and transform the Gaussian's
LOCAL parameters (offset, rotation, scale) into WORLD space:

    k        = mean edge length of the triangle           (scalar, per triangle)
    b        = triangle centroid                          (3,)  world origin
    R        = [t1, t2, n] orthonormal frame              (3,3) from edges+normal
    mu_world = k * R @ dmu + b                             (position)
    R_world  = R @ r                                       (rotation; r local)
    s_world  = k * s                                       (scale)

Only the LOCAL params (dmu, r, s, opacity, SH color) are optimized; the triangle
frame (b, R, k) comes from the posed FLAME mesh each frame. This is standard
morphable-model math — reimplemented independently (GaussianAvatars is Toyota
non-commercial; we vendor none of their code). See docs/LICENSES.md.

Reference: GaussianAvatars, Qian et al., CVPR 2024 (arXiv:2312.02069), §3.2.

Conventions:
    * Rotations are unit quaternions in (w, x, y, z) order, matching the NIVA
      export format (gaussian_format.py).
    * Tensors are batched as (..., N, D); works unbatched (N, D) too.
    * Pure PyTorch, differentiable, device-agnostic — CPU-testable without a GPU.
"""

from __future__ import annotations

import torch


# ---------------------------------------------------------------------------
# Triangle frames
# ---------------------------------------------------------------------------

def compute_triangle_frames(verts: torch.Tensor, faces: torch.Tensor):
    """Compute per-triangle (centroid b, rotation R, scale k).

    Args:
        verts: (..., V, 3) mesh vertices (posed FLAME mesh for a frame).
        faces: (F, 3) long triangle vertex indices (shared across the batch).

    Returns:
        b: (..., F, 3) triangle centroids.
        R: (..., F, 3, 3) orthonormal frames with columns [t1, t2, n].
        k: (..., F) per-triangle scale = mean edge length.

    Frame construction (deterministic, differentiable):
        v0, v1, v2 = triangle vertices
        e1 = v1 - v0 ;  e2 = v2 - v0
        n  = normalize(e1 x e2)                (triangle normal)
        t1 = normalize(e1)                      (first tangent)
        t2 = n x t1                             (second tangent, completes RH frame)
        R  = [t1 | t2 | n]  (columns)
        b  = (v0 + v1 + v2) / 3
        k  = mean(|e1|, |e2|, |v2 - v1|)        (mean edge length)
    """
    if faces.dtype not in (torch.int64, torch.int32, torch.long):
        faces = faces.long()

    v0 = verts.index_select(-2, faces[:, 0])
    v1 = verts.index_select(-2, faces[:, 1])
    v2 = verts.index_select(-2, faces[:, 2])

    e1 = v1 - v0
    e2 = v2 - v0
    e3 = v2 - v1

    eps = 1e-12
    n = torch.cross(e1, e2, dim=-1)
    n = n / (n.norm(dim=-1, keepdim=True) + eps)
    t1 = e1 / (e1.norm(dim=-1, keepdim=True) + eps)
    t2 = torch.cross(n, t1, dim=-1)
    t2 = t2 / (t2.norm(dim=-1, keepdim=True) + eps)

    # R columns = [t1, t2, n]  -> R @ x maps local axes to world
    R = torch.stack([t1, t2, n], dim=-1)  # (..., F, 3, 3)

    b = (v0 + v1 + v2) / 3.0
    k = (e1.norm(dim=-1) + e2.norm(dim=-1) + e3.norm(dim=-1)) / 3.0  # (..., F)

    return b, R, k


# ---------------------------------------------------------------------------
# Quaternion helpers (w, x, y, z)
# ---------------------------------------------------------------------------

def quat_normalize(q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return q / (q.norm(dim=-1, keepdim=True) + eps)


def quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """Unit quaternion (..., 4) in (w,x,y,z) -> rotation matrix (..., 3, 3)."""
    q = quat_normalize(q)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    tx, ty, tz = 2 * x, 2 * y, 2 * z
    twx, twy, twz = tx * w, ty * w, tz * w
    txx, txy, txz = tx * x, ty * x, tz * x
    tyy, tyz, tzz = ty * y, tz * y, tz * z
    o = torch.stack([
        1 - (tyy + tzz), txy - twz, txz + twy,
        txy + twz, 1 - (txx + tzz), tyz - twx,
        txz - twy, tyz + twx, 1 - (txx + tyy),
    ], dim=-1)
    return o.reshape(q.shape[:-1] + (3, 3))


def rotmat_to_quat(Rm: torch.Tensor) -> torch.Tensor:
    """Rotation matrix (..., 3, 3) -> unit quaternion (..., 4) in (w,x,y,z).

    Numerically-stable branchless-ish construction (handles all trace cases).
    """
    m = Rm
    m00, m11, m22 = m[..., 0, 0], m[..., 1, 1], m[..., 2, 2]
    trace = m00 + m11 + m22

    # Four candidate quaternions (one per largest-component branch); pick stable.
    q0 = torch.stack([
        1.0 + trace,
        m[..., 2, 1] - m[..., 1, 2],
        m[..., 0, 2] - m[..., 2, 0],
        m[..., 1, 0] - m[..., 0, 1],
    ], dim=-1)
    q1 = torch.stack([
        m[..., 2, 1] - m[..., 1, 2],
        1.0 + m00 - m11 - m22,
        m[..., 0, 1] + m[..., 1, 0],
        m[..., 0, 2] + m[..., 2, 0],
    ], dim=-1)
    q2 = torch.stack([
        m[..., 0, 2] - m[..., 2, 0],
        m[..., 0, 1] + m[..., 1, 0],
        1.0 - m00 + m11 - m22,
        m[..., 1, 2] + m[..., 2, 1],
    ], dim=-1)
    q3 = torch.stack([
        m[..., 1, 0] - m[..., 0, 1],
        m[..., 0, 2] + m[..., 2, 0],
        m[..., 1, 2] + m[..., 2, 1],
        1.0 - m00 - m11 + m22,
    ], dim=-1)

    cond0 = trace > 0
    cond1 = (m00 >= m11) & (m00 >= m22)
    cond2 = m11 >= m22
    q = torch.where(cond0[..., None], q0,
                    torch.where(cond1[..., None], q1,
                                torch.where(cond2[..., None], q2, q3)))
    return quat_normalize(q)


# ---------------------------------------------------------------------------
# Binding transform (Eq. 1)
# ---------------------------------------------------------------------------

def bind_gaussians(
    local_offset: torch.Tensor,     # (..., N, 3)  dmu
    local_quat: torch.Tensor,       # (..., N, 4)  r (w,x,y,z)
    local_scale: torch.Tensor,      # (..., N, 3)  s (linear)
    parent_idx: torch.Tensor,       # (N,) long -> triangle index per Gaussian
    b: torch.Tensor,                # (..., F, 3)
    R: torch.Tensor,                # (..., F, 3, 3)
    k: torch.Tensor,                # (..., F)
):
    """Transform LOCAL Gaussian params to WORLD via parent-triangle frames (Eq.1).

    Returns:
        world_pos:   (..., N, 3)
        world_quat:  (..., N, 4) in (w,x,y,z)
        world_scale: (..., N, 3) linear
    """
    if parent_idx.dtype not in (torch.int64, torch.int32, torch.long):
        parent_idx = parent_idx.long()

    # Gather parent triangle frame for each Gaussian along the F axis (dim -1/-2/-3)
    bR = R.index_select(-3, parent_idx)       # (..., N, 3, 3)
    bb = b.index_select(-2, parent_idx)       # (..., N, 3)
    bk = k.index_select(-1, parent_idx)       # (..., N)
    bk1 = bk.unsqueeze(-1)                     # (..., N, 1)

    # position:  mu_world = k * R @ dmu + b
    world_pos = bk1 * torch.einsum("...ij,...j->...i", bR, local_offset) + bb

    # rotation:  R_world = R @ r  -> quaternion
    r_mat = quat_to_rotmat(local_quat)         # (..., N, 3, 3)
    world_mat = torch.einsum("...ij,...jk->...ik", bR, r_mat)
    world_quat = rotmat_to_quat(world_mat)

    # scale:  s_world = k * s
    world_scale = bk1 * local_scale

    return world_pos, world_quat, world_scale
