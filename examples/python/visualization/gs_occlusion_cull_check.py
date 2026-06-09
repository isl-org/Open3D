#!/usr/bin/env python3
# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------
"""
Utility: measure Hi-Z occlusion culling efficiency on a Gaussian splat scene.

Renders 5 frames with a camera dolly (translation along Z), printing
GaussianGpuCounters for each frame.  Runs twice: once without and once with
occlusion culling.  The ratio (culled entries / baseline entries) quantifies
how much composite work the culling saves.

Usage:
    python gs_occlusion_cull_check.py [path_to.ply]

Default scene:
    /home/ssheorey/open3d_data/download/3dgs_example_assets/
        mipnerf360_garden_crop_table.ply
"""

import sys
import numpy as np
import open3d as o3d
import open3d.visualization.rendering as rendering

# ── Configuration ─────────────────────────────────────────────────────────────
PLY_PATH = (sys.argv[1] if len(sys.argv) > 1 else
            "/home/ssheorey/Downloads/models/garden/point_cloud/iteration_30000/point_cloud.ply")
WIDTH, HEIGHT = 1920, 1080
NUM_FRAMES = 6
# Camera dolly: step along the camera's forward axis each frame
DOLLY_STEP = 0.3  # metres

START_VIEW = {
	"eye" : 
	[
		-2.6868054866790771,
		0.99044573307037354,
		-0.63225322961807251
	],
	"far_plane" : 534.86212158203125,
	"field_of_view" : 60.0,
	"height" : 2766,
	"lookat" : 
	[
		13.334591865539551,
		-0.79696989059448242,
		8.7038984298706055
	],
	"near_plane" : 1.0,
	"up" : 
	[
		0.34091445803642273,
		-0.89126735925674438,
		-0.29903146624565125
	],
	"width" : 4608
}


# renderer = None
# ── Helpers ───────────────────────────────────────────────────────────────────
def make_renderer(occlusion_cull: bool) -> rendering.OffscreenRenderer:
    """Create an OffscreenRenderer with or without occlusion culling."""
    # global renderer
    # if renderer is None:
    renderer = rendering.OffscreenRenderer(WIDTH, HEIGHT)
    # else:
    #     renderer.scene.remove_geometry("splat")
    gsplat = o3d.t.io.read_point_cloud(PLY_PATH)

    mat = rendering.MaterialRecord()
    mat.shader = "gaussianSplat"
    mat.gaussian_splat_sh_degree = 2
    mat.gaussian_splat_occlusion_cull = occlusion_cull

    renderer.scene.add_geometry("splat", gsplat, mat)
    return renderer


def run_experiment(occlusion_cull: bool) -> list[dict]:
    """Render NUM_FRAMES frames and return per-frame counter dicts."""
    label = "WITH occlusion culling" if occlusion_cull else "WITHOUT occlusion culling"
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  {'Frame':>5}  {'splat_count':>12}  {'total_entries':>14}  "
          f"{'tile_count':>11}  {'error_flags':>12}")
    print(f"  {'-'*5}  {'-'*12}  {'-'*14}  {'-'*11}  {'-'*12}")

    render = make_renderer(occlusion_cull)

    # Initialize camera from START_VIEW dictionary when available.
    bbox = render.scene.bounding_box
    extent = bbox.get_extent()
    depth = float(np.linalg.norm(extent))

    eye = np.array(START_VIEW.get("eye", [0.0, 0.0, depth * 0.8]), dtype=float)
    centre = np.array(START_VIEW.get("lookat", bbox.get_center()), dtype=float)
    up = np.array(START_VIEW.get("up", [0.0, 1.0, 0.0]), dtype=float)
    fov = float(START_VIEW.get("field_of_view", 45.0))
    near = float(START_VIEW.get("near_plane", 0.1))
    far = float(START_VIEW.get("far_plane", max(depth * 5.0, 1.0)))

    # Compute forward direction vector for dolly
    forward = centre - eye
    forward_norm = forward / np.linalg.norm(forward)

    results = []
    for frame in range(NUM_FRAMES):
        # Set up camera view for the current frame
        render.setup_camera(
            fov,  # vertical FoV
            centre.tolist(),  # look-at target
            eye.tolist(),  # eye position
            up.tolist(),  # up vector
            near_clip=near,
            far_clip=far,
        )

        # render_to_image triggers the full geometry + composite pipeline
        render.render_to_image()
        c = render.get_gs_frame_counters()
        results.append(c)

        print(
            f"  {frame:>5}  {c['splat_count']:>12,}  {c['total_entries']:>14,}  "
            f"{c['tile_count']:>11,}  {c['error_flags']:>12}")

        if frame + 1 < NUM_FRAMES:
            # Dolly forward along the calculated look direction
            eye = eye + forward_norm * DOLLY_STEP
            centre = centre + forward_norm * DOLLY_STEP

    return results


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    print(f"Scene : {PLY_PATH}")
    print(f"Size  : {WIDTH}×{HEIGHT}")

    baseline = run_experiment(occlusion_cull=False)
    culled = run_experiment(occlusion_cull=True)

    # Summary: skip frame 0 for culled (no prior frame available yet)
    print(f"\n{'='*60}")
    print("  Summary (frames 1-{}, after warm-up)".format(NUM_FRAMES - 1))
    print(f"{'='*60}")
    print(f"  {'Frame':>5}  {'baseline_entries':>17}  {'culled_entries':>15}  "
          f"{'saved_%':>8}  {'baseline_splats':>16}  {'culled_splats':>14}")
    print(f"  {'-'*5}  {'-'*17}  {'-'*15}  {'-'*8}  {'-'*16}  {'-'*14}")

    total_base = 0
    total_cull = 0
    for i in range(1, NUM_FRAMES):
        b_entries = baseline[i]["total_entries"]
        c_entries = culled[i]["total_entries"]
        b_splats = baseline[i]["splat_count"]
        c_splats = culled[i]["splat_count"]
        pct = (1.0 - c_entries / b_entries) * 100.0 if b_entries > 0 else 0.0
        total_base += b_entries
        total_cull += c_entries
        print(f"  {i:>5}  {b_entries:>17,}  {c_entries:>15,}  "
              f"{pct:>7.1f}%  {b_splats:>16,}  {c_splats:>14,}")

    overall_pct = ((1.0 - total_cull / total_base) *
                   100.0 if total_base > 0 else 0.0)
    print(
        f"\n  Overall tile-entry reduction: {overall_pct:.1f}%  "
        f"({total_base - total_cull:,} entries saved across {NUM_FRAMES-1} frames)"
    )

    if baseline[0]["splat_count"] == 0:
        print(
            "\n  WARNING: splat_count=0 — Gaussian splat rendering may not be "
            "active (Vulkan required; CPU fallback does not support compute).")


if __name__ == "__main__":
    main()
