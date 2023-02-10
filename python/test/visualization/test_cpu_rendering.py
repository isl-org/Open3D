# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2023 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import platform
import os
from multiprocessing import Process
import pytest


def draw_box_offscreen():
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


@pytest.mark.skipif(
    not (platform.system() == "Linux" and platform.machine() == "x86_64") or
    os.getenv("OPEN3D_CPU_RENDERING", '') != 'true',
    reason="Offscreen CPU rendering is only supported on x86_64 Linux")
def test_draw_cpu():
    """Test CPU rendering in a separate process."""
    proc = Process(target=draw_box_offscreen)
    proc.start()
    proc.join(timeout=5)  # Wait for process to complete
    if proc.exitcode is None:
        proc.kill()  # Kill on failure
        assert False, __name__ + " did not complete."
    assert proc.exitcode == 0
