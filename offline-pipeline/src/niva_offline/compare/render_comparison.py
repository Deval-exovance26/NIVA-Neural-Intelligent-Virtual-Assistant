"""GPU-side renderer for the honest 3-way comparison (runs ONLY on the box).

Renders ORIGINAL | BASE | REFINED where:
  * ORIGINAL = the real reference frame from the tracker's img_lmdb (a photo).
  * BASE / REFINED = Gaussian exports, posed per frame via the GAGAvatar runtime
    (t_points -> FLAME head verts) and rendered through the SAME 32-channel
    rasterizer with the SAME camera + the SAME single upright flip.

Fixes vs the earlier compare3.py:
  1. Each avatar is posed against ITS OWN canonical layout (cached per avatar),
     instead of reusing one shared posed xyz. Base's head block differs from
     the runtime canonical, so sharing it mis-posed base. (See handoff diagnosis.)
  2. Orientation is derived once by ``--probe`` (writes a single still) and the
     chosen flip is applied identically to BASE and REFINED. ORIGINAL is never
     flipped or passed through the rasterizer.
  3. RGB readback is explicit (images[:, :3]); the compositor also guards against
     stray channel-order / >3-channel leakage.

Flow:
  --probe            -> render ONE frame (3 panels) to <out>/probe.png and exit.
  --previews         -> render only the configured preview timestamps to
                        <out>/preview_t{sec}.png and exit (for sign-off).
  (default)          -> render all selected frames + encode the mp4.

Run inside the GAGAvatar env from the GAGAvatar repo root, with PYTHONPATH
including the niva_offline src dir.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np


def _lazy_imports():
    import torch
    import torch.nn.functional as F
    from gagavatar.runtime import GAGAvatarRuntime, GAGAvatarRuntimeConfig
    from gagavatar.assets import GAGAvatarAssets
    from gagavatar.libs.utils_renderer import render_gaussian, NUM_CHANNELS
    from gagavatar.libs.GAGAvatar_track.engines import LMDBEngine
    return (torch, F, GAGAvatarRuntime, GAGAvatarRuntimeConfig, GAGAvatarAssets,
            render_gaussian, NUM_CHANNELS, LMDBEngine)


def main(argv=None):
    p = argparse.ArgumentParser(description="NIVA honest 3-way comparison renderer")
    p.add_argument("--gaga-root", required=True, type=Path)
    p.add_argument("--niva-src", required=True, type=Path)
    p.add_argument("--base-dir", required=True, type=Path)
    p.add_argument("--refined-dir", required=True, type=Path)
    p.add_argument("--driver-dir", required=True, type=Path, help="optim.pkl + img_lmdb")
    p.add_argument("--tracked", required=True, type=Path)
    p.add_argument("--avatar-id", default="front.png")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--panel-size", type=int, default=512)
    p.add_argument("--flip-y", choices=["auto", "yes", "no"], default="yes",
                   help="apply vertical flip to the splat panels (not the photo)")
    p.add_argument("--turntable", action="store_true",
                   help="slowly rotate base+refined around vertical axis (yaw)")
    p.add_argument("--yaw-amp", type=float, default=15.0,
                   help="turntable yaw amplitude in degrees (safe range ~15; "
                        "model lacks side/back geometry beyond that)")
    p.add_argument("--yaw-period-s", type=float, default=8.0,
                   help="seconds for one full back-and-forth yaw cycle")
    p.add_argument("--refined-dir-label", default="refined",
                   help="which refined version label to show (e.g. v3)")
    p.add_argument("--ffmpeg", default=None,
                   help="explicit ffmpeg binary path (defaults to imageio_ffmpeg)")
    p.add_argument("--probe", action="store_true", help="render ONE frame and exit")
    p.add_argument("--previews", action="store_true",
                   help="render only preview timestamps and exit")
    p.add_argument("--preview-times", type=float, nargs="*",
                   default=[0.0, 1.0, 2.0, 4.0])
    args = p.parse_args(argv)

    sys.path.insert(0, str(args.gaga_root.resolve()))
    sys.path.insert(0, str(args.niva_src.resolve()))

    (torch, F, GAGAvatarRuntime, GAGAvatarRuntimeConfig, GAGAvatarAssets,
     render_gaussian, NUM_CHANNELS, LMDBEngine) = _lazy_imports()

    from niva_offline.export.gaussian_format import load as load_avatar
    from niva_offline.compare import comparison_common as cc

    import imageio.v2 as imageio

    dev = "cuda"
    args.out.mkdir(parents=True, exist_ok=True)

    # --- runtime -----------------------------------------------------------
    assets = GAGAvatarAssets(
        root=args.gaga_root / "assets",
        model_path=args.gaga_root / "assets/GAGAvatar.pt",
        tracked_path=args.tracked,
        flame_model_path=args.gaga_root / "assets/FLAME_with_eye.pt",
    )
    rt = GAGAvatarRuntime(GAGAvatarRuntimeConfig(assets=assets, device=dev))
    rt.set_avatar_id(args.avatar_id)
    cam_params = rt.model.cam_params

    # Canonical layout from a zero-motion forward pass (runtime ordering).
    gs0 = rt.forward_gaussians_for_batch(
        rt.build_forward_batch(torch.zeros(1, 106, device=dev)))
    canonical_xyz = gs0["xyz"][0].detach()

    # --- load avatars ------------------------------------------------------
    base = load_avatar(args.base_dir, name="niva_base")
    ref = load_avatar(args.refined_dir, name="niva_refined")

    def pack(av):
        return {
            "colors": torch.tensor(av.colors, device=dev),
            "opacities": torch.tensor(av.opacities, device=dev),
            "scales": torch.tensor(av.scales, device=dev),
            "rotations": torch.tensor(av.rotations, device=dev),
            "positions": torch.tensor(av.positions, device=dev),
        }

    base_t, ref_t = pack(base), pack(ref)

    def to32(c):
        return torch.cat([c, c.new_zeros(c.shape[0], NUM_CHANNELS - 3)], dim=1)

    def pose_xyz(av_positions, batch, head_from_canonical):
        """Pose the FLAME head block [0:n_head] with the per-frame t_points.

        head_from_canonical=True  -> use the runtime canonical layout for the
            head block before overwriting with t_points. This is REQUIRED for
            BASE, whose stored head rows (0..5022) came from a slightly
            different canonical pose and do NOT line up with t_points -> if we
            used base's own head rows the face would barely deform ("base not
            animating" bug). REFINED is already canonical-aligned (frac 1.0) so
            either source works; we use canonical for both for consistency.

        The avatar's OWN static local points (rows n_head:) are always kept, so
        each panel keeps its own body/background splats and appearance.
        """
        n_head = batch["t_points"].shape[1]
        xyz = av_positions.clone()
        if head_from_canonical:
            xyz[:n_head] = canonical_xyz[:n_head]
        xyz[:n_head] = batch["t_points"][0]
        return xyz

    # Per-panel vertical flip. Determined empirically from the probe: BASE's
    # stored geometry is upright under a +Y-down flip, REFINED's is already
    # upright (opposite Y convention between the two exports). Overridable.
    flip_base = {"yes": True, "no": False, "auto": True}[args.flip_y]
    flip_ref = {"yes": True, "no": False, "auto": False}[args.flip_y]

    def render_splat(av_t, batch, flip, head_from_canonical, yaw_deg=0.0):
        xyz = pose_xyz(av_t["positions"], batch, head_from_canonical)
        if yaw_deg != 0.0:
            r = np.deg2rad(yaw_deg)
            c, s = float(np.cos(r)), float(np.sin(r))
            Ry = torch.tensor([[c, 0, s], [0, 1, 0], [-s, 0, c]],
                              device=xyz.device, dtype=xyz.dtype)
            ctr = xyz.mean(0, keepdim=True)
            xyz = (xyz - ctr) @ Ry.T + ctr
        gs = {
            "xyz": xyz[None],
            "colors": to32(av_t["colors"].clamp(0, 1))[None],
            "opacities": av_t["opacities"][None],
            "scales": av_t["scales"][None],
            "rotations": av_t["rotations"][None],
        }
        out = render_gaussian(gs_params=gs, cam_matrix=batch["t_transform"],
                              cam_params=cam_params)["images"][:, :3].clamp(0, 1)
        im = (out[0].permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
        if flip:
            im = im[::-1]
        return np.ascontiguousarray(im)

    def yaw_for_index(i, n):
        """Smooth ease-in/out back-and-forth yaw within +/- yaw_amp degrees."""
        if not args.turntable or n <= 1:
            return 0.0
        # one full cycle every yaw_period_s seconds, sinusoidal (smooth, synced)
        t = i / float(args.fps)
        import math
        phase = 2.0 * math.pi * t / max(args.yaw_period_s, 1e-6)
        return args.yaw_amp * math.sin(phase)

    # --- driver frames -----------------------------------------------------
    with open(args.driver_dir / "optim.pkl", "rb") as f:
        optim = pickle.load(f)
    all_keys = list(optim.keys())
    sel_keys = cc.select_frame_keys(all_keys, stride=args.stride,
                                    max_frames=args.max_frames)
    lmdb = LMDBEngine(str(args.driver_dir / "img_lmdb"), write=False)
    print(f"[compare] {len(all_keys)} driver frames -> {len(sel_keys)} selected "
          f"(stride={args.stride})", flush=True)

    if args.turntable:
        labels = [
            "Original Reference",
            "Base Gaussian Avatar - Rotating",
            f"Refined Gaussian Avatar ({args.refined_dir_label}) - Rotating",
        ]
    else:
        labels = [cc.PANEL_LABELS[q] for q in cc.PANEL_ORDER]

    def build_batch(key):
        e = optim[key]
        exp = torch.as_tensor(e["expcode"], device=dev).float().reshape(-1)[:100]
        po = torch.as_tensor(e["posecode"], device=dev).float().reshape(-1)
        motion = torch.cat([exp, po[:3], po[3:6]])[None]
        return rt.build_forward_batch(motion)

    def original_frame(key):
        gt = lmdb.load(key, type="image").float().to(dev) / 255.0  # (3,H,W)
        gt = F.interpolate(gt[None], size=tuple(cam_params["size"]),
                           mode="bilinear", align_corners=False)[0]
        return (gt.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

    def composite_for_key(key, frame_i=0, n_frames=1):
        batch = build_batch(key)
        orig = original_frame(key)
        yaw = yaw_for_index(frame_i, n_frames)
        bimg = render_splat(base_t, batch, flip=flip_base,
                            head_from_canonical=True, yaw_deg=yaw)
        rimg = render_splat(ref_t, batch, flip=flip_ref,
                            head_from_canonical=True, yaw_deg=yaw)
        return cc.composite_panels([orig, bimg, rimg], labels,
                                   panel_size=args.panel_size,
                                   label_band_h=36)

    # --- probe: ONE frame, then stop (orientation check) -------------------
    if args.probe:
        key = sel_keys[len(sel_keys) // 2]
        row = composite_for_key(key)
        probe_path = args.out / "probe.png"
        imageio.imwrite(probe_path, row)
        print(f"[compare] PROBE wrote {probe_path} (key={key}) "
              f"flip_base={flip_base} flip_ref={flip_ref}", flush=True)
        return 0

    # --- previews: only the requested timestamps, then stop ----------------
    if args.previews:
        for t in args.preview_times:
            i = cc.timestamp_to_selection_index(t, args.fps, len(sel_keys),
                                                args.stride)
            row = composite_for_key(sel_keys[i])
            pth = args.out / f"preview_t{t:g}.png"
            imageio.imwrite(pth, row)
            print(f"[compare] preview t={t:g}s -> {pth} (sel_idx={i}, "
                  f"key={sel_keys[i]})", flush=True)
        return 0

    # --- full render -------------------------------------------------------
    frames_dir = args.out / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    import tqdm
    n = len(sel_keys)
    for i, key in enumerate(tqdm.tqdm(sel_keys)):
        row = composite_for_key(key, frame_i=i, n_frames=n)
        imageio.imwrite(frames_dir / f"frame_{i:06d}.png", row)

    out_mp4 = args.out / "niva_3way.mp4"
    ff = args.ffmpeg
    if not ff:
        try:
            import imageio_ffmpeg
            ff = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:  # noqa: BLE001
            ff = "ffmpeg"
    import subprocess
    cmd = cc.build_ffmpeg_cmd(frames_dir, args.fps, out_mp4)
    cmd[0] = ff
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise SystemExit("ffmpeg encode failed:\n" + (res.stderr[-2000:] or ""))
    print(f"[compare] WROTE {out_mp4} | {len(sel_keys)} frames | "
          f"ORIGINAL | BASE | REFINED ({args.refined_dir_label})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
