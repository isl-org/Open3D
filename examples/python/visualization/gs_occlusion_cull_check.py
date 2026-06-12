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
import threading
import time
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


# ── Helpers ───────────────────────────────────────────────────────────────────
import open3d.visualization.gui as gui

def print_summary(baseline, culled) -> None:
    # Summary of culling efficiency
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

    if len(baseline) > 0 and baseline[0]["splat_count"] == 0:
        print(
            "\n  WARNING: splat_count=0 — Gaussian splat rendering may not be "
            "active (Vulkan required; CPU fallback does not support compute).")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    print(f"Scene : {PLY_PATH}")
    print(f"Size  : {WIDTH}×{HEIGHT}")

    gsplat = o3d.t.io.read_point_cloud(PLY_PATH)
    # Set up GUI Application and Window
    app = gui.Application.instance
    app.initialize()

    window = app.create_window("Gaussian Splat Occlusion Culling Check", WIDTH, HEIGHT)
    scene_widget = gui.SceneWidget()
    window.add_child(scene_widget)

    # Callback on windows resize or layout
    def on_layout(layout_context):
        scene_widget.frame = window.content_rect
    window.set_on_layout(on_layout)

    # Initialize scene and Gaussian splat geometry
    scene_widget.scene = rendering.Open3DScene(window.renderer)
    scene_widget.enable_scene_caching(False)  # Redraw dynamically without caching

    mat = rendering.MaterialRecord()
    mat.shader = "gaussianSplat"
    mat.gaussian_splat_sh_degree = 2
    mat.gaussian_splat_occlusion_cull = True  # Start without occlusion culling
    scene_widget.scene.add_geometry("splat", gsplat, mat)

    # Camera setup (matching starting view)
    fov = float(START_VIEW.get("field_of_view", 60.0))
    near = float(START_VIEW.get("near_plane", 1.0))
    far = float(START_VIEW.get("far_plane", 534.86212158203125))
    scene_widget.scene.camera.set_projection(
        fov,
        float(WIDTH) / float(HEIGHT),
        near,
        far,
        rendering.Camera.FovType.Vertical
    )

    baseline_results = []
    culled_results = []

    # Experiment run inside background thread to coordinate frame steps safely
    def run_experiment():
        time.sleep(0.5)  # Let window fully open

        for run in range(2):
            occlusion_cull = (run == 0)
            label = "WITH occlusion culling" if occlusion_cull else "WITHOUT occlusion culling"

            # Update material property
            def update_material():
                mat.gaussian_splat_occlusion_cull = occlusion_cull
                scene_widget.scene.modify_geometry_material("splat", mat)
            app.post_to_main_thread(window, update_material)
            time.sleep(0.05)

            print(f"\n{'='*60}")
            print(f"  {label}")
            print(f"{'='*60}")
            print(f"  {'Frame':>5}  {'splat_count':>12}  {'total_entries':>14}  "
                  f"{'tile_count':>11}  {'error_flags':>12}")
            print(f"  {'-'*5}  {'-'*12}  {'-'*14}  {'-'*11}  {'-'*12}")

            eye = np.array(START_VIEW["eye"], dtype=float)
            centre = np.array(START_VIEW["lookat"], dtype=float)
            up = np.array(START_VIEW["up"], dtype=float)
            forward = centre - eye
            forward_norm = forward / np.linalg.norm(forward)

            for frame in range(NUM_FRAMES):
                current_eye = eye.tolist()
                current_centre = centre.tolist()
                current_up = up.tolist()

                # 1. Update camera view on the main thread and trigger redraw
                def set_view():
                    scene_widget.scene.camera.look_at(current_centre, current_eye, current_up)
                    scene_widget.force_redraw()
                app.post_to_main_thread(window, set_view)

                # Wait for the frame to be drawn completely on the GPU
                time.sleep(0.12)

                # 2. Get frame counters on the main thread
                counters = [None]
                def get_counters():
                    counters[0] = window.renderer.get_gs_frame_counters()
                app.post_to_main_thread(window, get_counters)

                # Wait for counters retrieval
                time.sleep(0.04)
                c = counters[0]
                if occlusion_cull:
                    culled_results.append(c)
                else:
                    baseline_results.append(c)

                print(
                    f"  {frame:>5}  {c['splat_count']:>12,}  {c['total_entries']:>14,}  "
                    f"{c['tile_count']:>11,}  {c['error_flags']:>12}")

                # Dolly forward
                if frame + 1 < NUM_FRAMES:
                    eye += forward_norm * DOLLY_STEP
                    centre += forward_norm * DOLLY_STEP

        # Finish up and output comparison
        def wrap_up():
            window.close()
            print_summary(baseline_results, culled_results)
        app.post_to_main_thread(window, wrap_up)

    threading.Thread(target=run_experiment).start()
    app.run()


if __name__ == "__main__":
    main()
