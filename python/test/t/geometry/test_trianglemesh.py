# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# The MIT License (MIT)
#
# Copyright (c) 2018-2021 www.open3d.org
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
# ----------------------------------------------------------------------------

from turtle import width
import open3d as o3d
import open3d.core as o3c
import numpy as np
import pytest


def test_clip_plane():
    cube = o3d.t.geometry.TriangleMesh.from_legacy(
        o3d.geometry.TriangleMesh.create_box())
    clipped_cube = cube.clip_plane(point=[0.5, 0, 0], normal=[1, 0, 0])
    assert clipped_cube.vertex['positions'].shape == (12, 3)
    assert clipped_cube.triangle['indices'].shape == (14, 3)


def test_create_box():
    # test with default parameters
    box_default = o3d.t.geometry.TriangleMesh.create_box()

    vertex_positions_default = o3c.Tensor(np.array([[0.0, 0.0, 0.0],
                                                    [1.0, 0.0, 0.0],
                                                    [0.0, 0.0, 1.0],
                                                    [1.0, 0.0, 1.0],
                                                    [0.0, 1.0, 0.0],
                                                    [1.0, 1.0, 0.0],
                                                    [0.0, 1.0, 1.0],
                                                    [1.0, 1.0, 1.0]]),
                                          dtype=o3c.Dtype.Float64,
                                          device=o3c.Device("CPU:0"))

    triangle_indices = o3c.Tensor(np.array([[4, 7, 5], [4, 6, 7], [0, 2, 4],
                                            [2, 6, 4], [0, 1, 2], [1, 3, 2],
                                            [1, 5, 7], [1, 7, 3], [2, 3, 7],
                                            [2, 7, 6], [0, 4, 1], [1, 4, 5]]),
                                  dtype=o3c.Dtype.Int64,
                                  device=o3c.Device("CPU:0"))

    assert box_default.vertex['positions'].allclose(vertex_positions_default)
    assert box_default.triangle['indices'].allclose(triangle_indices)

    # test with custom parameters
    box_custom = o3d.t.geometry.TriangleMesh.create_box(2, 4, 3)

    vertex_positions_custom = o3c.Tensor(np.array([[0.0, 0.0, 0.0],
                                                   [2.0, 0.0, 0.0],
                                                   [0.0, 0.0, 3.0],
                                                   [2.0, 0.0, 3.0],
                                                   [0.0, 4.0, 0.0],
                                                   [2.0, 4.0, 0.0],
                                                   [0.0, 4.0, 3.0],
                                                   [2.0, 4.0, 3.0]]),
                                         dtype=o3c.Dtype.Float64,
                                         device=o3c.Device("CPU:0"))

    assert box_custom.vertex['positions'].allclose(vertex_positions_custom)
    assert box_custom.triangle['indices'].allclose(triangle_indices)
