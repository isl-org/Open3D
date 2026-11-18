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
from multiprocessing import Process, get_context
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


def draw_untextured_mesh_offscreen():
    """Checks that an untextured legacy mesh has initialized fallback UVs."""
    import numpy as np
    import open3d as o3d
    import open3d.visualization.rendering as rendering

    # Construct the renderer first to preserve the allocation order that made
    # the unwritten UV bytes visible with glibc's MALLOC_PERTURB_.
    render = rendering.OffscreenRenderer(1920, 1080)
    render.scene.set_background([0.58, 0.58, 0.58, 1.0])
    render.scene.show_skybox(False)
    render.scene.view.set_post_processing(True)
    render.scene.view.set_antialiasing(True)
    render.scene.view.set_ambient_occlusion(False)
    render.scene.view.set_shadowing(False, rendering.View.ShadowType.VSM)

    cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=0.5,
                                                         height=1.5,
                                                         resolution=6,
                                                         split=6)
    cylinder.translate([0.0, 0.0, 0.75])
    cylinder.compute_vertex_normals()
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(cylinder.vertices).copy()),
        o3d.utility.Vector3iVector(np.asarray(cylinder.triangles).copy()))
    mesh.compute_vertex_normals()

    material = rendering.MaterialRecord()
    material.shader = "defaultLit"
    material.base_color = [0.263, 0.117, 0.552, 1.0]
    material.base_roughness = 0.55
    render.scene.add_geometry("mesh", mesh, material)
    render.scene.scene.set_sun_light([-0.4, -0.6, -1.0], [1.0, 1.0, 1.0], 75000)
    render.scene.scene.enable_sun_light(True)
    render.setup_camera(45.0, [0.0, 0.0, 0.4], [2.3, -2.3, 1.8],
                        [0.0, 0.0, 1.0])

    pixels = np.asarray(render.render_to_image())
    bright_pixels = np.all(pixels[:, :, :3] > 200, axis=2).sum()
    assert bright_pixels < 1000


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
def test_untextured_mesh_cpu(monkeypatch):
    """Test fallback UV initialization with deterministic allocator fill."""
    monkeypatch.setenv("MALLOC_PERTURB_", "1")
    proc = get_context("spawn").Process(target=draw_untextured_mesh_offscreen)
    proc.start()
    proc.join(timeout=30)
    if proc.exitcode is None:
        proc.kill()
        proc.join()
        pytest.fail(__name__ + " did not complete.")
    assert proc.exitcode == 0
