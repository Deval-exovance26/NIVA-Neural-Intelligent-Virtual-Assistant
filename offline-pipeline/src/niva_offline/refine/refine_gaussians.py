"""Task 8 / Priority 3 — per-subject Gaussian refinement against the reference video.

Optimizes the per-Gaussian APPEARANCE parameters of the exported NIVA avatar
(colors, opacity, and optionally scale) so the rendered avatar better matches the
real female reference frames — fixing the Stage-1 frontal-only color limitation
(off-axis color, mouth/teeth region) while PRESERVING identity.

Design (honest about what is/ isn't optimized):
    * Positions are FLAME-driven per frame (via the GAGAvatar runtime's
      build_forward_batch -> t_points). We do NOT free-optimize xyz, so the
      head geometry/identity stays locked to FLAME. This is the identity-
      preserving guarantee.
    * We optimize: colors (N,3), opacity logits (N,1), log-scale residual (N,3).
      Scale is regularized (log-space clamp) to avoid spiky Gaussians.
    * Rendering reuses GAGAvatar's compiled 32-channel rasterizer by padding
      RGB -> 32ch and reading back the first 3 channels (validated by _probe).
    * Loss = L1(image, gt) + L1 on the face bbox crop (sharpens the face) +
      opacity/scale regularizers.

Reference frames come from the tracker's img_lmdb + optim.pkl (the same driver
used for inference), so each optimized frame has a true photo to match.

Checkpointing: every --ckpt-every iters we save optimizable tensors +
iteration + optimizer state to <out>/ckpt.pt (resumable with --resume), and
export the current avatar in the browser format to <out>/refined so the demo /
view.html can show progress.

Run inside GAGAvatar env from the GAGAvatar repo root (PYTHONPATH=repo root).
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def inverse_sigmoid(x, eps=1e-6):
    x = x.clamp(eps, 1 - eps)
    return torch.log(x / (1 - x))


def main(argv=None):
    p = argparse.ArgumentParser(description="NIVA Gaussian refinement (Task 8)")
    p.add_argument("--niva-src", required=True, type=Path)
    p.add_argument("--export-dir", required=True, type=Path, help="input baked export dir")
    p.add_argument("--driver", required=True, type=Path, help="tracker output dir (img_lmdb+optim.pkl)")
    p.add_argument("--tracked", required=True, type=Path, help="tracked.pt (avatar features)")
    p.add_argument("--avatar-id", default="front.png")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--iters", type=int, default=10000)
    p.add_argument("--ckpt-every", type=int, default=1000)
    p.add_argument("--preview-every", type=int, default=1000)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--anchor", type=float, default=5.0,
                   help="weight anchoring colors/opacity to the base (prevents speckle)")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--name", default="niva_refined")
    args = p.parse_args(argv)

    sys.path.insert(0, str(args.niva_src.resolve()))
    from niva_offline.export.gaussian_format import GaussianAvatar, load, save

    dev = "cuda"
    from gagavatar.runtime import GAGAvatarRuntime, GAGAvatarRuntimeConfig
    from gagavatar.assets import GAGAvatarAssets
    from gagavatar.libs.utils_renderer import render_gaussian, NUM_CHANNELS
    from gagavatar.libs.GAGAvatar_track.engines import LMDBEngine

    args.out.mkdir(parents=True, exist_ok=True)

    # --- runtime + driver frames ------------------------------------------
    assets = GAGAvatarAssets(
        root=Path("./assets"), model_path=Path("./assets/GAGAvatar.pt"),
        tracked_path=args.tracked, flame_model_path=Path("./assets/FLAME_with_eye.pt"),
    )
    runtime = GAGAvatarRuntime(GAGAvatarRuntimeConfig(assets=assets, device=dev))
    runtime.set_avatar_id(args.avatar_id)

    with open(args.driver / "optim.pkl", "rb") as f:
        optim = pickle.load(f)
    frame_keys = sorted(optim.keys(), key=lambda k: int(k.split("_")[-1]))
    lmdb = LMDBEngine(str(args.driver / "img_lmdb"), write=False)
    print(f"[refine] {len(frame_keys)} driver frames")

    # --- load baked export as optimizable appearance ----------------------
    av = load(args.export_dir, name="niva_base")
    N = av.num_gaussians
    base_scales = torch.tensor(av.scales, device=dev)
    rotations = torch.tensor(av.rotations, device=dev)

    colors = torch.tensor(av.colors, device=dev).clamp(1e-4, 1 - 1e-4)
    base_colors = colors.clone()                    # frozen anchor (known-good identity)
    colors = colors.clone().requires_grad_(True)
    base_opacity = torch.tensor(av.opacities, device=dev).clamp(1e-4, 1 - 1e-4)
    opacity_logit = inverse_sigmoid(torch.tensor(av.opacities, device=dev)).requires_grad_(True)
    log_scale_res = torch.zeros_like(base_scales).requires_grad_(True)  # multiplicative residual

    opt = torch.optim.Adam([
        {"params": [colors], "lr": args.lr},
        {"params": [opacity_logit], "lr": args.lr},
        {"params": [log_scale_res], "lr": args.lr * 0.3},
    ], betas=(0.9, 0.99))

    start_iter = 0
    ckpt_path = args.out / "ckpt.pt"
    if args.resume and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=dev)
        colors.data = ck["colors"]; opacity_logit.data = ck["opacity_logit"]
        log_scale_res.data = ck["log_scale_res"]; opt.load_state_dict(ck["opt"])
        start_iter = ck["iter"]
        print(f"[refine] resumed from iter {start_iter}")

    cam_params = runtime.model.cam_params

    def build_frame(key):
        """Return (gt_image(3,H,W) in [0,1], t_transform(3,4), t_points(N_flame,3))."""
        entry = optim[key]
        exp = torch.as_tensor(entry["expcode"], device=dev).float().reshape(-1)[:100]
        pose = torch.as_tensor(entry["posecode"], device=dev).float().reshape(-1)
        motion = torch.cat([exp, pose[:3], pose[3:6]])[None]
        batch = runtime.build_forward_batch(motion)
        gt = lmdb.load(key, type="image").float().to(dev) / 255.0  # (3,H,W)
        gt = F.interpolate(gt[None], size=tuple(cam_params["size"]), mode="bilinear",
                           align_corners=False)[0]
        return gt, batch

    def current_xyz(batch, cached_xyz):
        # positions: FLAME verts (first 5023) posed per frame + fixed local points
        xyz = cached_xyz.clone()
        xyz[:batch["t_points"].shape[1]] = batch["t_points"][0]
        return xyz

    # cache base xyz layout from a forward_gaussians call
    gs0 = runtime.forward_gaussians_for_batch(runtime.build_forward_batch(
        torch.zeros(1, 106, device=dev)))
    cached_xyz = gs0["xyz"][0].detach()

    def to32(c):
        return torch.cat([c, c.new_zeros(c.shape[0], NUM_CHANNELS - 3)], dim=1)

    def render(batch, xyz):
        gs = {
            "xyz": xyz[None],
            "colors": to32(colors.clamp(0, 1))[None],
            "opacities": torch.sigmoid(opacity_logit)[None],
            "scales": (base_scales * torch.exp(log_scale_res).clamp(0.2, 5.0))[None],
            "rotations": rotations[None],
        }
        out = render_gaussian(gs_params=gs, cam_matrix=batch["t_transform"], cam_params=cam_params)
        return out["images"][:, :3].clamp(0, 1)  # (1,3,H,W)

    def export_current(tag):
        avatar = GaussianAvatar(
            positions=cached_xyz.cpu().numpy(),
            rotations=rotations.cpu().numpy(),
            scales=(base_scales * torch.exp(log_scale_res).clamp(0.2, 5.0)).detach().cpu().numpy(),
            opacities=torch.sigmoid(opacity_logit).detach().cpu().numpy(),
            colors=colors.detach().clamp(0, 1).cpu().numpy(),
            meta={"source": f"refined @ iter {tag}", "subject": "niva-female",
                  "base": "niva_base", "note": "Task 8 appearance refinement"},
        )
        save(avatar, args.out / "refined", name=args.name)

    rng = np.random.default_rng(0)
    t0 = time.time()
    for it in range(start_iter, args.iters):
        key = frame_keys[int(rng.integers(len(frame_keys)))]
        gt, batch = build_frame(key)
        xyz = current_xyz(batch, cached_xyz)
        pred = render(batch, xyz)
        img_loss = F.l1_loss(pred[0], gt)
        # Anchor to the known-good base appearance so refinement makes only small,
        # coherent corrections (prevents the free-RGB speckle failure mode where
        # the raw 3-ch splat overfits the UNet-rendered target). Identity is in
        # the base colors; we stay close to them.
        color_anchor = F.mse_loss(colors.clamp(0, 1), base_colors)
        opacity_anchor = F.mse_loss(torch.sigmoid(opacity_logit), base_opacity)
        reg = (args.anchor * color_anchor
               + args.anchor * opacity_anchor
               + 1e-3 * log_scale_res.pow(2).mean())
        loss = img_loss + reg

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if it % 50 == 0:
            psnr = -10 * torch.log10(F.mse_loss(pred[0], gt).detach() + 1e-8)
            rate = (it - start_iter + 1) / (time.time() - t0)
            print(f"[refine] it={it} loss={loss.item():.4f} img={img_loss.item():.4f} "
                  f"anchor={color_anchor.item():.5f} psnr={psnr.item():.2f} {rate:.1f}it/s", flush=True)

        if (it + 1) % args.ckpt_every == 0 or it + 1 == args.iters:
            torch.save({"colors": colors.data, "opacity_logit": opacity_logit.data,
                        "log_scale_res": log_scale_res.data, "opt": opt.state_dict(),
                        "iter": it + 1}, ckpt_path)
            export_current(it + 1)
            print(f"[refine] checkpoint+export @ it={it+1} -> {args.out/'refined'}", flush=True)

    export_current(args.iters)
    print("[OK] refinement complete ->", args.out / "refined")


if __name__ == "__main__":
    raise SystemExit(main())
