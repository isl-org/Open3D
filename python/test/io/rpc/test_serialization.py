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

import os
import open3d as o3d
import numpy as np
import pytest


def test_set_mesh_data_deserialization():
    """Tests the deserialization of messages created with the set_mesh_data_function.
    """

    def set_mesh_data_to_geometry(*args, **kwargs):
        bc = o3d.io.rpc.BufferConnection()
        o3d.io.rpc.set_mesh_data(*args, **kwargs, connection=bc)
        return o3d.io.rpc.data_buffer_to_meta_geometry(bc.get_buffer())

    rng = np.random.RandomState(123)

    verts = rng.rand(100, 3).astype(np.float32)
    tris = rng.randint(0, 100, size=[71, 3]).astype(np.int32)
    lines = rng.randint(0, 100, size=[82, 2]).astype(np.int32)

    dtypes = [np.uint8, np.int16, np.int32, np.float32, np.float64]
    vert_attrs = {
        'a':
            rng.uniform(0, 256, size=verts.shape).astype(rng.choice(dtypes)),
        'b':
            rng.uniform(0, 256,
                        size=verts.shape[:1] + (7,)).astype(rng.choice(dtypes)),
        'c':
            rng.uniform(0, 256,
                        size=verts.shape + (2,)).astype(rng.choice(dtypes)),
    }

    tri_attrs = {
        'a':
            rng.uniform(0, 256,
                        size=tris.shape[:1] + (1,)).astype(rng.choice(dtypes)),
        'b':
            rng.uniform(0, 256,
                        size=tris.shape[:1] + (3,)).astype(rng.choice(dtypes)),
        'c':
            rng.uniform(0, 256,
                        size=tris.shape[:1] + (5,)).astype(rng.choice(dtypes)),
    }

    line_attrs = {
        'a':
            rng.uniform(0, 256,
                        size=lines.shape[:1] + (1,)).astype(rng.choice(dtypes)),
        'b':
            rng.uniform(0, 256, size=lines.shape[:1] + (1, 2, 3)).astype(
                rng.choice(dtypes)),
        'c':
            rng.uniform(0, 256,
                        size=lines.shape[:1] + (2,)).astype(rng.choice(dtypes)),
    }

    path, time, geom = set_mesh_data_to_geometry(verts,
                                                 vertex_attributes=vert_attrs,
                                                 path="pcd",
                                                 time=123)
    assert path == "pcd"
    assert time == 123
    assert isinstance(geom, o3d.t.geometry.PointCloud)
    np.testing.assert_equal(geom.point['positions'].numpy(), verts)
    for k, v in vert_attrs.items():
        np.testing.assert_equal(geom.point[k].numpy(), v)

    path, time, geom = set_mesh_data_to_geometry(verts,
                                                 faces=tris,
                                                 vertex_attributes=vert_attrs,
                                                 face_attributes=tri_attrs,
                                                 path="trimesh",
                                                 time=123)
    assert path == "trimesh"
    assert time == 123
    assert isinstance(geom, o3d.t.geometry.TriangleMesh)
    np.testing.assert_equal(geom.vertex['positions'].numpy(), verts)
    np.testing.assert_equal(geom.triangle['indices'].numpy(), tris)
    for k, v in vert_attrs.items():
        np.testing.assert_equal(geom.vertex[k].numpy(), v)
    for k, v in tri_attrs.items():
        np.testing.assert_equal(geom.triangle[k].numpy(), v)

    path, time, geom = set_mesh_data_to_geometry(verts,
                                                 lines=lines,
                                                 vertex_attributes=vert_attrs,
                                                 line_attributes=line_attrs,
                                                 path="lines",
                                                 time=123)
    assert path == "lines"
    assert time == 123
    assert isinstance(geom, o3d.t.geometry.LineSet)
    np.testing.assert_equal(geom.point['positions'].numpy(), verts)
    np.testing.assert_equal(geom.line['indices'].numpy(), lines)
    for k, v in vert_attrs.items():
        np.testing.assert_equal(geom.point[k].numpy(), v)
    for k, v in line_attrs.items():
        np.testing.assert_equal(geom.line[k].numpy(), v)
