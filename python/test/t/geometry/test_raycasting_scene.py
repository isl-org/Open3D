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

import open3d as o3d
import numpy as np
import pytest


# test intersection with a single triangle
def test_cast_rays():

    vertices = o3d.core.Tensor([[0, 0, 0], [1, 0, 0], [1, 1, 0]],
                               dtype=o3d.core.Dtype.Float32)
    triangles = o3d.core.Tensor([[0, 1, 2]], dtype=o3d.core.Dtype.UInt32)

    scene = o3d.t.geometry.RaycastingScene()
    geom_id = scene.add_triangles(vertices, triangles)

    rays = o3d.core.Tensor([[0.2, 0.1, 1, 0, 0, -1], [10, 10, 10, 1, 0, 0]],
                           dtype=o3d.core.Dtype.Float32)
    ans = scene.cast_rays(rays)

    # first ray hits the triangle
    assert geom_id == ans['geometry_ids'][0]
    assert np.isclose(ans['t_hit'][0].item(), 1.0)

    # second ray misses
    assert o3d.t.geometry.RaycastingScene.INVALID_ID == ans['geometry_ids'][1]
    assert np.isinf(ans['t_hit'][1].item())


# cast lots of random rays to test the internal batching
# we expect no errors for this test
def test_cast_lots_of_rays():
    vertices = o3d.core.Tensor([[0, 0, 0], [1, 0, 0], [1, 1, 0]],
                               dtype=o3d.core.Dtype.Float32)
    triangles = o3d.core.Tensor([[0, 1, 2]], dtype=o3d.core.Dtype.UInt32)

    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(vertices, triangles)

    rs = np.random.RandomState(123)
    rays = o3d.core.Tensor.from_numpy(rs.rand(7654321, 6).astype(np.float32))

    _ = scene.cast_rays(rays)