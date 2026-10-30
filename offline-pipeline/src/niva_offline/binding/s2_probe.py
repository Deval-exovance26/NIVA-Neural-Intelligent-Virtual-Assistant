"""S2 probe (run ON THE BOX, GAGAvatar env): render the standalone SH splat.

Gets a posed FLAME mesh + camera from the GAGAvatar runtime, builds one SH
Gaussian per FLAME triangle (S1 binding), renders through gsplat's SH rasterizer
(NO StyleUNet), and writes a frontal still so we can confirm the standalone splat
is upright + recognisable before refinement (S3).

Usage (on box):
  export CUDA_HOME=/usr/local/cuda-12.8
  export PATH=~/miniconda3/envs/GAGAvatar/bin:$CUDA_HOME/bin:$PATH
  python -m niva_offline.binding.s2_probe --gaga-root <GAGA> --driver <DRV> \
      --out ~/_s2_probe --sh-degree 3
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--gaga-root", required=True, type=Path)
    p.add_argument("--niva-src", required=True, type=Path)
    p.add_argument("--driver", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--sh-degree", type=int, default=3)
    p.add_argument("--avatar-id", default="front.png")
    args = p.parse_args(argv)

    sys.path.insert(0, str(args.gaga_root.resolve()))
    sys.path.insert(0, str(args.niva_src.resolve()))

    import imageio.v2 as imageio
    from gsplat import rasterization
    from gagavatar.runtime import GAGAvatarRuntime, GAGAvatarRuntimeConfig
    from gagavatar.assets import GAGAvatarAssets
    from gagavatar.libs.GAGAvatar_track.engines import LMDBEngine

    from niva_offline.binding.standalone_splat import (
        init_splat_from_faces, world_gaussians, sh_dc_to_rgb,
    )

    dev = "cuda"
    args.out.mkdir(parents=True, exist_ok=True)

    assets = GAGAvatarAssets(
        root=args.gaga_root / "assets",
        model_path=args.gaga_root / "assets/GAGAvatar.pt",
        tracked_path=args.gaga_root / "render_results/tracked/tracked.pt",
        flame_model_path=args.gaga_root / "assets/FLAME_with_eye.pt",
    )
    rt = GAGAvatarRuntime(GAGAvatarRuntimeConfig(assets=assets, device=dev))
    rt.set_avatar_id(args.avatar_id)
    cam_params = rt.model.cam_params
    H, W = int(cam_params["size"][0]), int(cam_params["size"][1])
    fx = float(cam_params["focal_x"]) * W / 2.0
    fy = float(cam_params["focal_y"]) * H / 2.0
    K = torch.tensor([[[fx, 0, W / 2.0], [0, fy, H / 2.0], [0, 0, 1.0]]], device=dev)

    # FLAME faces
    flame = torch.load(args.gaga_root / "assets/FLAME_with_eye.pt",
                       map_location="cpu", weights_only=False)
    faces = flame["flame_model"]["f"].long()

    # A posed neutral frame + its camera from the driver
    with open(args.driver / "optim.pkl", "rb") as f:
        optim = pickle.load(f)
    keys = sorted(optim, key=lambda k: int(k.split("_")[-1]))
    key = keys[0]
    e = optim[key]
    exp = torch.as_tensor(e["expcode"], device=dev).float().reshape(-1)[:100]
    po = torch.as_tensor(e["posecode"], device=dev).float().reshape(-1)
    batch = rt.build_forward_batch(torch.cat([exp, po[:3], po[3:6]])[None])
    verts = batch["t_points"][0]          # (5023,3) posed FLAME verts
    cam = batch["t_transform"][0]          # (3,4) [R|t] world->cam

    # Sample an initial per-triangle colour from the real frame (frontal) so the
    # init isn't flat gray: project triangle centroids, sample the image.
    lmdb = LMDBEngine(str(args.driver / "img_lmdb"), write=False)
    img = (lmdb.load(key, type="image").float().to(dev) / 255.0)  # (3,H,W)
    import torch.nn.functional as Fn
    img = Fn.interpolate(img[None], size=(H, W), mode="bilinear",
                         align_corners=False)[0]

    splat = init_splat_from_faces(faces, verts, sh_degree=args.sh_degree,
                                  device=dev)
    pos, quat, scale = world_gaussians(splat, verts)
    opac = torch.sigmoid(splat.opacity_logit)

    # Build gsplat viewmat (world->cam, 4x4). GAGAvatar cam is [R|t]; try the
    # INRIA-style x/y flip to match its image convention. Validated by eye.
    def viewmat_from(cam3x4, flip_xy):
        V = torch.eye(4, device=dev)
        V[:3, :3] = cam3x4[:3, :3]
        V[:3, 3] = cam3x4[:3, 3]
        if flip_xy:
            V[:2, :] *= -1.0
        return V[None]

    gsplat_quats = quat[:, [0, 1, 2, 3]]  # gsplat uses (w,x,y,z) too

    for tag, flip in [("flipxy", True), ("noflip", False)]:
        V = viewmat_from(cam, flip)
        out, alpha, _ = rasterization(
            pos, gsplat_quats, scale, opac, splat.sh, V, K, width=W, height=H,
            sh_degree=args.sh_degree, render_mode="RGB",
        )
        im = (out[0].clamp(0, 1).detach().cpu().numpy() * 255).astype(np.uint8)
        imageio.imwrite(args.out / f"s2_{tag}.png", im)
        print(f"[s2] wrote s2_{tag}.png  range {im.min()}..{im.max()} "
              f"mean {im.mean():.1f}", flush=True)

    # Also dump the real frame for side-by-side.
    gt = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    imageio.imwrite(args.out / "s2_gt.png", gt)
    print(f"[s2] N gaussians = {splat.num_gaussians}, sh_degree={args.sh_degree}")
    print("[s2] DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
