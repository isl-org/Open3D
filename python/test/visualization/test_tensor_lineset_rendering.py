# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2026 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------
"""Rendering tests for fixed-topology tensor LineSet updates."""

import os
import platform
from multiprocessing import Process

import numpy as np
import pytest


def update_tensor_line_set_offscreen():
    """Updates a fixed-topology tensor line set without recreating it."""
    import open3d as o3d
    import open3d.visualization.rendering as rendering

    points = np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]],
                      dtype=np.float32)
    lines = np.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=np.int64)
    colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]],
                      dtype=np.float32)
    line_set = o3d.t.geometry.LineSet(o3d.core.Tensor(points),
                                      o3d.core.Tensor(lines))
    line_set.line.colors = o3d.core.Tensor(colors)
    render = rendering.OffscreenRenderer(320, 240)
    material = rendering.MaterialRecord()
    material.shader = "unlitLine"
    material.line_width = 4.0
    render.scene.add_geometry("lines",
                              line_set,
                              material,
                              add_downsampled_copy_for_fast_rendering=False)
    render.setup_camera(60.0, [0, 0, 0], [0, 0, 3], [0, 1, 0])
    render.scene.view.set_post_processing(False)
    before = np.asarray(render.render_to_image()).copy()

    points[:, 0] += 0.5
    line_set.point.positions = o3d.core.Tensor(points)
    line_set.line.colors = o3d.core.Tensor(1.0 - colors)
    render.scene.update_geometry(
        "lines", line_set,
        rendering.Scene.UPDATE_POINTS_FLAG | rendering.Scene.UPDATE_COLORS_FLAG)
    after = np.asarray(render.render_to_image()).copy()

    bounds = render.scene.bounding_box
    assert np.isclose(bounds.min_bound[0], -0.5)
    assert np.isclose(bounds.max_bound[0], 1.5)
    assert np.any(before != after)

    render.scene.remove_geometry("lines")
    render.scene.add_geometry("lines",
                              line_set,
                              material,
                              add_downsampled_copy_for_fast_rendering=False)
    rebuilt = np.asarray(render.render_to_image()).copy()
    assert np.array_equal(after, rebuilt)


@pytest.mark.skipif(
    not (platform.system() == "Linux" and platform.machine() == "x86_64") or
    os.getenv("OPEN3D_CPU_RENDERING", '') != 'true',
    reason="Offscreen CPU rendering is only supported on x86_64 Linux")
def test_update_tensor_line_set_cpu():
    """Tensor line updates reuse thin or wide line buffers."""
    proc = Process(target=update_tensor_line_set_offscreen)
    proc.start()
    proc.join(timeout=10)
    if proc.exitcode is None:
        proc.kill()
        proc.join()
        pytest.fail(__name__ + " did not complete.")
    assert proc.exitcode == 0
