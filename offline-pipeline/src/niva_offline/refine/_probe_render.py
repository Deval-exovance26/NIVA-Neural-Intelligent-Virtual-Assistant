"""Probe: render our baked-RGB Gaussians through GAGAvatar's 32-ch rasterizer
by padding RGB -> 32 channels, and confirm gradients flow to RGB.

Run inside GAGAvatar env from the GAGAvatar repo root.
"""
import sys
from pathlib import Path
import torch

def main():
    niva_src = sys.argv[1]
    export_dir = sys.argv[2]
    sys.path.insert(0, niva_src)
    from niva_offline.export.gaussian_format import load
    from gagavatar.libs.utils_renderer import render_gaussian, NUM_CHANNELS

    dev = "cuda"
    av = load(export_dir, name="niva_base")
    N = av.num_gaussians
    xyz = torch.tensor(av.positions, device=dev)
    scales = torch.tensor(av.scales, device=dev)
    rots = torch.tensor(av.rotations, device=dev)
    opac = torch.tensor(av.opacities, device=dev)
    rgb = torch.tensor(av.colors, device=dev, requires_grad=True)

    print("NUM_CHANNELS =", NUM_CHANNELS)

    # pad rgb (N,3) -> (N,32)
    def to32(c):
        pad = c.new_zeros(c.shape[0], NUM_CHANNELS - 3)
        return torch.cat([c, pad], dim=1)

    # identity-ish camera: look down -Z at the avatar center
    center = xyz.mean(0)
    cam = torch.eye(4, device=dev)[:3]  # (3,4)
    cam[:3, 3] = torch.tensor([center[0], center[1], center[2] + 1.0], device=dev) * -1.0
    cam = cam[None]

    gs = {
        "xyz": xyz[None], "colors": to32(rgb)[None], "opacities": opac[None],
        "scales": scales[None], "rotations": rots[None],
    }
    cam_params = {"focal_x": 12.0, "focal_y": 12.0, "size": [512, 512]}
    out = render_gaussian(gs_params=gs, cam_matrix=cam, cam_params=cam_params)
    img = out["images"]  # (1,32,512,512)
    print("render out shape:", tuple(img.shape))
    rgb_img = img[:, :3]
    print("rgb_img range:", float(rgb_img.min()), float(rgb_img.max()), "mean:", float(rgb_img.mean()))

    # fake loss on first 3 channels -> grad to rgb
    loss = rgb_img.mean()
    loss.backward()
    print("rgb.grad is not None:", rgb.grad is not None, "grad abs mean:", float(rgb.grad.abs().mean()))
    print("PROBE_OK")

if __name__ == "__main__":
    main()
