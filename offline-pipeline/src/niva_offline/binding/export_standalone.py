"""Export a standalone SH splat to the browser .bin/.json flat-RGB format.

Bakes the SH DC band -> flat RGB (the current browser renderer is flat-RGB; SH
shaders arrive in S5). Poses the splat at a chosen neutral frame via the S1
binding and writes positions/rotations/scales/opacities/colors.

Can export either the S2 INIT splat (no ckpt) or an S3 refinement checkpoint
(--ckpt). Used to publish progressive previews to the web demo across S2->S5.

Run in GAGAvatar env with gsplat available.
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
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--name", default="niva_standalone")
    p.add_argument("--sh-degree", type=int, default=3)
    p.add_argument("--splats-per-face", type=int, default=1)
    p.add_argument("--scale-clip", type=float, default=0.04,
                   help="clamp max world splat scale for browser (0=off)")
    p.add_argument("--ckpt", type=Path, default=None, help="S3 ckpt.pt (optional)")
    p.add_argument("--avatar-id", default="front.png")
    p.add_argument("--stage-label", default="S2")
    p.add_argument("--flip-y", action="store_true",
                   help="negate position Y to match browser -Y-up convention")
    args = p.parse_args(argv)

    sys.path.insert(0, str(args.gaga_root.resolve()))
    sys.path.insert(0, str(args.niva_src.resolve()))

    from gagavatar.runtime import GAGAvatarRuntime, GAGAvatarRuntimeConfig
    from gagavatar.assets import GAGAvatarAssets
    from niva_offline.binding.standalone_splat import (
        init_splat_from_faces, world_gaussians, sh_dc_to_rgb,
    )
    from niva_offline.export.gaussian_format import GaussianAvatar, save

    dev = "cuda"
    assets = GAGAvatarAssets(
        root=args.gaga_root / "assets",
        model_path=args.gaga_root / "assets/GAGAvatar.pt",
        tracked_path=args.gaga_root / "render_results/tracked/tracked.pt",
        flame_model_path=args.gaga_root / "assets/FLAME_with_eye.pt",
    )
    rt = GAGAvatarRuntime(GAGAvatarRuntimeConfig(assets=assets, device=dev))
    rt.set_avatar_id(args.avatar_id)

    flame = torch.load(args.gaga_root / "assets/FLAME_with_eye.pt",
                       map_location="cpu", weights_only=False)
    faces = flame["flame_model"]["f"].long().to(dev)

    with open(args.driver / "optim.pkl", "rb") as f:
        optim = pickle.load(f)
    keys = sorted(optim, key=lambda k: int(k.split("_")[-1]))
    key = keys[0]
    e = optim[key]
    exp = torch.as_tensor(e["expcode"], device=dev).float().reshape(-1)[:100]
    po = torch.as_tensor(e["posecode"], device=dev).float().reshape(-1)
    batch = rt.build_forward_batch(torch.cat([exp, po[:3], po[3:6]])[None])
    verts = batch["t_points"][0].detach()

    splat = init_splat_from_faces(faces, verts, sh_degree=args.sh_degree,
                                  splats_per_face=args.splats_per_face, device=dev)

    if args.ckpt is not None and args.ckpt.exists():
        s = torch.load(args.ckpt, map_location=dev)
        splat.sh = s["sh"]
        splat.opacity_logit = s["opacity_logit"]
        splat.local_scale = torch.exp(s["log_scale"])
        splat.local_offset = s["local_offset"]
        print(f"[export] loaded ckpt iter {s.get('iter')}")

    pos, quat, scale = world_gaussians(splat, verts)
    opac = torch.sigmoid(splat.opacity_logit).reshape(-1, 1)
    rgb = sh_dc_to_rgb(splat.sh[:, 0, :])  # DC band -> flat RGB

    # Browser-safe clamp: the WebGL2/WebGPU renderer uses order-independent
    # blending without the EWA/sorting gsplat does. A handful of oversized splats
    # (world scale up to ~0.9) blow up into huge quads/spikes. Cap world scale to
    # a sane ceiling so the browser render matches the on-box gsplat look.
    if args.scale_clip > 0:
        scale = scale.clamp(max=args.scale_clip)

    pos_np = pos.detach().cpu().numpy()
    quat_np = quat.detach().cpu().numpy()
    if args.flip_y:
        # Mirror about the X-Z plane to match the browser's -Y-up convention
        # (the existing base/refined assets use that convention). Flipping Y of
        # positions; for the quaternion (w,x,y,z) a reflection isn't a rotation,
        # so we instead rotate 180deg about X (negates Y and Z of positions) —
        # but we only want Y. Simplest robust approach: negate position Y and
        # mirror the rotation's x,z components so splats keep facing out.
        pos_np = pos_np.copy(); pos_np[:, 1] *= -1.0
        quat_np = quat_np.copy()
        quat_np[:, 2] *= -1.0  # mirror y-rotation axis to keep surface facing

    avatar = GaussianAvatar(
        positions=pos_np,
        rotations=quat_np,
        scales=scale.detach().cpu().numpy(),
        opacities=opac.detach().cpu().numpy(),
        colors=rgb.detach().cpu().numpy(),
        meta={"source": f"standalone splat ({args.stage_label})",
              "subject": "niva-female", "sh_degree": args.sh_degree,
              "flip_y": bool(args.flip_y),
              "note": "SH DC baked to flat RGB for current browser renderer"},
    )
    save(avatar, args.out_dir, name=args.name)
    print(f"[export] wrote {args.out_dir}/{args.name}.{{bin,json}}  "
          f"N={avatar.num_gaussians}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
