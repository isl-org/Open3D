# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2026 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------
"""Tests CPU (software) offscreen rendering for the new Filament-based
open3d.visualization.rendering.OffscreenRenderer, used when no GPU is
available. See also test_legacy_headless_rendering.py, which covers the
legacy open3d.visualization.Visualizer's EGL offscreen fallback."""

import platform
import os
from multiprocessing import Process
import pytest


def draw_box_offscreen():
    """Runs in a separate process and renders a box with OffscreenRenderer."""
    import open3d as o3d
    import open3d.visualization.rendering as rendering
    render = rendering.OffscreenRenderer(640, 480)
    cube_red = o3d.geometry.TriangleMesh.create_box(1, 2, 4)
    cube_red.compute_vertex_normals()
    cube_red.paint_uniform_color((1.0, 0.0, 0.0))
    default_mat = rendering.MaterialRecord()
    render.scene.add_geometry("box", cube_red, default_mat)
    render.setup_camera(60.0, [0, 0, 0], [0, 10, 0], [0, 0, 1])
    _ = render.render_to_image()


def draw_tone_mapping_offscreen():
    """Checks that public tone-mapping selections reach Filament."""
    import numpy as np
    import open3d.visualization.rendering as rendering

    render = rendering.OffscreenRenderer(160, 90)
    scene = render.scene
    scene.show_skybox(False)
    scene.view.set_post_processing(True)

    grading = rendering.ColorGrading
    means = []
    for mode in (grading.ToneMapping.LINEAR, grading.ToneMapping.ACES):
        scene.view.set_color_grading(grading(grading.Quality.ULTRA, mode))
        scene.set_background([1.0, 1.0, 1.0, 1.0])
        pixels = np.asarray(render.render_to_image())
        means.append(pixels[15:75, 15:145, :3].mean())

    # LINEAR leaves the white background close to 255, while ACES compresses
    # it. The old code ignored both selections and differed only by dithering.
    assert abs(means[0] - means[1]) > 20.0


@pytest.mark.skipif(
    not (platform.system() == "Linux" and platform.machine() == "x86_64") or
    os.getenv("OPEN3D_CPU_RENDERING", '') != 'true',
    reason="Offscreen CPU rendering is only supported on x86_64 Linux")
def test_draw_cpu():
    """Test that OffscreenRenderer can render a box in a separate process."""
    proc = Process(target=draw_box_offscreen)
    proc.start()
    proc.join(timeout=5)  # Wait for process to complete
    if proc.exitcode is None:
        proc.kill()
        proc.join()  # Reap the killed process to avoid leaving a zombie.
        pytest.fail(__name__ + " did not complete.")
    assert proc.exitcode == 0


@pytest.mark.skipif(
    not (platform.system() == "Linux" and platform.machine() == "x86_64") or
    os.getenv("OPEN3D_CPU_RENDERING", '') != 'true',
    reason="Offscreen CPU rendering is only supported on x86_64 Linux")
def test_tone_mapping_cpu():
    """Test that supported tone-mapping modes produce different output."""
    proc = Process(target=draw_tone_mapping_offscreen)
    proc.start()
    proc.join(timeout=10)
    if proc.exitcode is None:
        proc.kill()
        proc.join()
        pytest.fail(__name__ + " did not complete.")
    assert proc.exitcode == 0
