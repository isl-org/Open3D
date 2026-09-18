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
import numpy as np
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


def update_tensor_mesh_offscreen():
    """Updates a fixed-topology tensor mesh without recreating the scene."""
    import open3d as o3d
    import open3d.visualization.rendering as rendering

    positions = np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]],
                         dtype=np.float32)
    triangles = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    mesh = o3d.t.geometry.TriangleMesh(o3d.core.Tensor(positions),
                                       o3d.core.Tensor(triangles))
    mesh.vertex.normals = o3d.core.Tensor(
        np.tile([0, 0, 1], (4, 1)).astype(np.float32))
    mesh.vertex.colors = o3d.core.Tensor(
        np.tile([1, 0, 0], (4, 1)).astype(np.float32))
    mesh.vertex.texture_uvs = o3d.core.Tensor(
        np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32))
    render = rendering.OffscreenRenderer(320, 240)
    material = rendering.MaterialRecord()
    material.shader = "defaultUnlit"
    render.scene.add_geometry("mesh",
                              mesh,
                              material,
                              add_downsampled_copy_for_fast_rendering=False)
    render.setup_camera(60.0, [0, 0, 0], [0, 0, 3], [0, 1, 0])
    before = np.asarray(render.render_to_image()).copy()

    positions[:, 0] += 0.5
    mesh.vertex.positions = o3d.core.Tensor(positions)
    mesh.vertex.normals = o3d.core.Tensor(
        np.tile([1, 0, 0], (4, 1)).astype(np.float32))
    mesh.vertex.colors = o3d.core.Tensor(
        np.tile([0, 1, 0], (4, 1)).astype(np.float32))
    mesh.vertex.texture_uvs = o3d.core.Tensor(
        np.array([[1, 1], [0, 1], [0, 0], [1, 0]], dtype=np.float32))
    render.scene.update_geometry(
        "mesh", mesh, rendering.Scene.UPDATE_POINTS_FLAG |
        rendering.Scene.UPDATE_NORMALS_FLAG |
        rendering.Scene.UPDATE_COLORS_FLAG | rendering.Scene.UPDATE_UV0_FLAG)
    after = np.asarray(render.render_to_image()).copy()

    bounds = render.scene.bounding_box
    assert np.isclose(bounds.min_bound[0], -0.5)
    assert np.isclose(bounds.max_bound[0], 1.5)
    assert np.any(before != after)


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
def test_update_tensor_mesh_cpu():
    """Tensor mesh updates reuse buffers and refresh scene bounds."""
    proc = Process(target=update_tensor_mesh_offscreen)
    proc.start()
    proc.join(timeout=10)
    if proc.exitcode is None:
        proc.kill()
        proc.join()
        pytest.fail(__name__ + " did not complete.")
    assert proc.exitcode == 0
