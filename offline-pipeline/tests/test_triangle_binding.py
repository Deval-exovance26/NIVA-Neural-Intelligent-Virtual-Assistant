"""Unit tests for the FLAME triangle binding core (S1). CPU-only, no GPU."""

from __future__ import annotations

import math

import pytest
import torch

from niva_offline.binding.triangle_binding import (
    bind_gaussians,
    compute_triangle_frames,
    quat_normalize,
    quat_to_rotmat,
    rotmat_to_quat,
)

torch.manual_seed(0)
IDENTITY_Q = torch.tensor([1.0, 0.0, 0.0, 0.0])


# ---- triangle frames -------------------------------------------------------

def test_frame_canonical_triangle():
    # Triangle in XY plane: v0=origin, v1=+X, v2=+Y.
    verts = torch.tensor([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]])
    faces = torch.tensor([[0, 1, 2]])
    b, R, k = compute_triangle_frames(verts, faces)
    # centroid = (1/3, 1/3, 0)
    assert torch.allclose(b[0], torch.tensor([1 / 3, 1 / 3, 0.0]), atol=1e-6)
    # t1 = +X (e1 normalized), n = +Z (e1 x e2), t2 = n x t1 = +Y
    assert torch.allclose(R[0, :, 0], torch.tensor([1.0, 0, 0]), atol=1e-6)  # t1
    assert torch.allclose(R[0, :, 2], torch.tensor([0.0, 0, 1]), atol=1e-6)  # n
    assert torch.allclose(R[0, :, 1], torch.tensor([0.0, 1, 0]), atol=1e-6)  # t2
    # R is a proper rotation
    assert torch.allclose(R[0] @ R[0].T, torch.eye(3), atol=1e-6)
    assert torch.allclose(torch.det(R[0]), torch.tensor(1.0), atol=1e-6)


def test_frame_scale_is_mean_edge_length():
    # Equilateral-ish: edges all length 2 -> k = 2.
    verts = torch.tensor([[0.0, 0, 0], [2, 0, 0], [1, math.sqrt(3), 0]])
    faces = torch.tensor([[0, 1, 2]])
    _, _, k = compute_triangle_frames(verts, faces)
    assert torch.allclose(k[0], torch.tensor(2.0), atol=1e-5)


def test_frame_translation_invariant_rotation_scale():
    verts = torch.tensor([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]])
    faces = torch.tensor([[0, 1, 2]])
    _, R0, k0 = compute_triangle_frames(verts, faces)
    _, R1, k1 = compute_triangle_frames(verts + 5.0, faces)
    assert torch.allclose(R0, R1, atol=1e-6)
    assert torch.allclose(k0, k1, atol=1e-6)


# ---- quaternion helpers ----------------------------------------------------

def test_quat_rotmat_roundtrip_random():
    q = quat_normalize(torch.randn(50, 4))
    Rm = quat_to_rotmat(q)
    q2 = rotmat_to_quat(Rm)
    # q and -q are the same rotation; compare rotation matrices instead.
    assert torch.allclose(quat_to_rotmat(q2), Rm, atol=1e-5)


def test_quat_identity_is_identity_matrix():
    assert torch.allclose(quat_to_rotmat(IDENTITY_Q), torch.eye(3), atol=1e-6)


def test_quat_to_rotmat_is_orthonormal():
    q = quat_normalize(torch.randn(10, 4))
    Rm = quat_to_rotmat(q)
    eye = torch.eye(3).expand(10, 3, 3)
    assert torch.allclose(torch.bmm(Rm, Rm.transpose(1, 2)), eye, atol=1e-5)


# ---- binding transform (Eq. 1) --------------------------------------------

def _single_triangle():
    verts = torch.tensor([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]])
    faces = torch.tensor([[0, 1, 2]])
    return compute_triangle_frames(verts, faces)


def test_bind_zero_offset_lands_on_centroid():
    b, R, k = _single_triangle()
    dmu = torch.zeros(1, 3)
    r = IDENTITY_Q[None]
    s = torch.ones(1, 3)
    parent = torch.tensor([0])
    pos, quat, scale = bind_gaussians(dmu, r, s, parent, b, R, k)
    assert torch.allclose(pos[0], b[0], atol=1e-6)  # offset 0 -> centroid


def test_bind_scale_multiplies_by_k():
    # Triangle with k=2 -> world scale = 2 * local scale.
    verts = torch.tensor([[0.0, 0, 0], [2, 0, 0], [1, math.sqrt(3), 0]])
    faces = torch.tensor([[0, 1, 2]])
    b, R, k = compute_triangle_frames(verts, faces)
    s = torch.tensor([[0.1, 0.2, 0.3]])
    pos, quat, scale = bind_gaussians(
        torch.zeros(1, 3), IDENTITY_Q[None], s, torch.tensor([0]), b, R, k)
    assert torch.allclose(scale[0], torch.tensor([0.2, 0.4, 0.6]), atol=1e-5)


def test_bind_offset_uses_triangle_frame_and_k():
    # Canonical triangle (k=1, R=I-ish): local +X offset of 0.5 -> world moves
    # by 0.5 along t1 (=+X), scaled by k.
    b, R, k = _single_triangle()  # k = mean edge length = (1 + 1 + sqrt2)/3
    dmu = torch.tensor([[0.5, 0.0, 0.0]])
    pos, _, _ = bind_gaussians(
        dmu, IDENTITY_Q[None], torch.ones(1, 3), torch.tensor([0]), b, R, k)
    expected = b[0] + k[0] * torch.tensor([0.5, 0.0, 0.0])
    assert torch.allclose(pos[0], expected, atol=1e-6)


def test_bind_rotation_composes():
    # local rotation about Z by 90deg, triangle frame = world axes (canonical).
    b, R, k = _single_triangle()
    ang = math.pi / 2
    rz = torch.tensor([[math.cos(ang / 2), 0.0, 0.0, math.sin(ang / 2)]])  # (w,x,y,z) z-rot
    pos, quat, _ = bind_gaussians(
        torch.zeros(1, 3), rz, torch.ones(1, 3), torch.tensor([0]), b, R, k)
    world_mat = quat_to_rotmat(quat[0])
    # R_frame (t1=X,t2=Y,n=Z) @ Rz  == Rz here since frame is identity-aligned
    expected = quat_to_rotmat(rz[0])
    assert torch.allclose(world_mat, expected, atol=1e-5)


def test_bind_rigid_motion_of_triangle_moves_gaussian_rigidly():
    # If the triangle translates+rotates, a bound Gaussian (fixed local params)
    # follows rigidly. Translate the whole mesh by t and check pos shifts by t.
    verts = torch.tensor([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]])
    faces = torch.tensor([[0, 1, 2]])
    dmu = torch.tensor([[0.3, -0.2, 0.1]])
    r = quat_normalize(torch.randn(1, 4))
    s = torch.rand(1, 3) + 0.1
    parent = torch.tensor([0])

    b0, R0, k0 = compute_triangle_frames(verts, faces)
    pos0, q0, sc0 = bind_gaussians(dmu, r, s, parent, b0, R0, k0)

    t = torch.tensor([3.0, -1.0, 2.0])
    b1, R1, k1 = compute_triangle_frames(verts + t, faces)
    pos1, q1, sc1 = bind_gaussians(dmu, r, s, parent, b1, R1, k1)

    assert torch.allclose(pos1[0] - pos0[0], t, atol=1e-5)       # translated
    assert torch.allclose(sc1, sc0, atol=1e-6)                    # scale unchanged
    assert torch.allclose(quat_to_rotmat(q1[0]), quat_to_rotmat(q0[0]), atol=1e-5)


def test_bind_is_differentiable():
    b, R, k = _single_triangle()
    dmu = torch.zeros(1, 3, requires_grad=True)
    r = IDENTITY_Q[None].clone().requires_grad_(True)
    s = torch.ones(1, 3, requires_grad=True)
    pos, quat, scale = bind_gaussians(dmu, r, s, torch.tensor([0]), b, R, k)
    (pos.sum() + scale.sum()).backward()
    assert dmu.grad is not None and s.grad is not None
    assert torch.isfinite(dmu.grad).all()


def test_bind_multiple_gaussians_multiple_triangles():
    verts = torch.tensor([
        [0.0, 0, 0], [1, 0, 0], [0, 1, 0],   # tri 0
        [5.0, 5, 5], [6, 5, 5], [5, 6, 5],   # tri 1
    ])
    faces = torch.tensor([[0, 1, 2], [3, 4, 5]])
    b, R, k = compute_triangle_frames(verts, faces)
    # 3 gaussians: two on tri0, one on tri1
    dmu = torch.zeros(3, 3)
    r = IDENTITY_Q[None].expand(3, 4).contiguous()
    s = torch.ones(3, 3)
    parent = torch.tensor([0, 0, 1])
    pos, _, _ = bind_gaussians(dmu, r, s, parent, b, R, k)
    assert torch.allclose(pos[0], b[0], atol=1e-6)
    assert torch.allclose(pos[1], b[0], atol=1e-6)
    assert torch.allclose(pos[2], b[1], atol=1e-6)  # near (5.33,5.33,5)
    assert pos[2][0] > 4  # clearly on the far triangle
