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

# RaycastingScene is not available for linux arm
pytestmark = pytest.mark.skipif(not hasattr(o3d.t.geometry, 'RaycastingScene'),
                                reason="RaycastingScene was not compiled")


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
    scene.add_triangles(vertices, triangles)

    rs = np.random.RandomState(123)
    rays = o3d.core.Tensor.from_numpy(rs.rand(7654321, 6).astype(np.float32))

    _ = scene.cast_rays(rays)


def test_add_triangle_mesh():
    cube = o3d.t.geometry.TriangleMesh.from_legacy_triangle_mesh(
        o3d.geometry.TriangleMesh.create_box())

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(cube)

    rays = o3d.core.Tensor([[0.5, 0.5, -1, 0, 0, 1], [0.5, 0.5, 0.5, 0, 0, 1],
                            [10, 10, 10, 1, 0, 0]],
                           dtype=o3d.core.Dtype.Float32)
    ans = scene.count_intersections(rays)

    assert (ans.numpy() == [2, 1, 0]).all()


def test_count_intersections():
    cube = o3d.t.geometry.TriangleMesh.from_legacy_triangle_mesh(
        o3d.geometry.TriangleMesh.create_box())

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(cube)

    rays = o3d.core.Tensor([[0.5, 0.5, -1, 0, 0, 1], [0.5, 0.5, 0.5, 0, 0, 1],
                            [10, 10, 10, 1, 0, 0]],
                           dtype=o3d.core.Dtype.Float32)
    ans = scene.count_intersections(rays)

    assert (ans.numpy() == [2, 1, 0]).all()


# count lots of random ray intersections to test the internal batching
# we expect no errors for this test
def test_count_lots_of_intersections():
    cube = o3d.t.geometry.TriangleMesh.from_legacy_triangle_mesh(
        o3d.geometry.TriangleMesh.create_box())

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(cube)

    rs = np.random.RandomState(123)
    rays = o3d.core.Tensor.from_numpy(rs.rand(1234567, 6).astype(np.float32))

    _ = scene.count_intersections(rays)


def test_compute_closest_points():
    vertices = o3d.core.Tensor([[0, 0, 0], [1, 0, 0], [1, 1, 0]],
                               dtype=o3d.core.Dtype.Float32)
    triangles = o3d.core.Tensor([[0, 1, 2]], dtype=o3d.core.Dtype.UInt32)

    scene = o3d.t.geometry.RaycastingScene()
    geom_id = scene.add_triangles(vertices, triangles)

    query_points = o3d.core.Tensor([[0.2, 0.1, 1], [10, 10, 10]],
                                   dtype=o3d.core.Dtype.Float32)
    ans = scene.compute_closest_points(query_points)

    assert (geom_id == ans['geometry_ids']).all()
    assert (0 == ans['primitive_ids']).all()
    assert np.allclose(ans['points'].numpy(),
                       np.array([[0.2, 0.1, 0.0], [1, 1, 0]]))


def test_compute_distance():
    cube = o3d.t.geometry.TriangleMesh.from_legacy_triangle_mesh(
        o3d.geometry.TriangleMesh.create_box())

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(cube)

    query_points = o3d.core.Tensor(
        [[0.5, 0.5, 0.5], [-0.5, -0.5, -0.5], [0, 0, 0]],
        dtype=o3d.core.Dtype.Float32)
    ans = scene.compute_distance(query_points)
    assert np.allclose(ans.numpy(), [0.5, np.sqrt(3 * 0.5**2), 0.0])


def test_compute_signed_distance():
    cube = o3d.t.geometry.TriangleMesh.from_legacy_triangle_mesh(
        o3d.geometry.TriangleMesh.create_box())

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(cube)

    query_points = o3d.core.Tensor(
        [[0.5, 0.5, 0.5], [-0.5, -0.5, -0.5], [0, 0, 0]],
        dtype=o3d.core.Dtype.Float32)
    ans = scene.compute_signed_distance(query_points)
    assert np.allclose(ans.numpy(), [-0.5, np.sqrt(3 * 0.5**2), 0.0])


def test_compute_occupancy():
    cube = o3d.t.geometry.TriangleMesh.from_legacy_triangle_mesh(
        o3d.geometry.TriangleMesh.create_box())

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(cube)

    query_points = o3d.core.Tensor([[0.5, 0.5, 0.5], [-0.5, -0.5, -0.5]],
                                   dtype=o3d.core.Dtype.Float32)
    ans = scene.compute_occupancy(query_points)
    assert np.allclose(ans.numpy(), [1.0, 0.0])


@pytest.mark.parametrize("shape", ([11], [1, 2, 3], [32, 14]))
def test_output_shapes(shape):
    vertices = o3d.core.Tensor([[0, 0, 0], [1, 0, 0], [1, 1, 0]],
                               dtype=o3d.core.Dtype.Float32)
    triangles = o3d.core.Tensor([[0, 1, 2]], dtype=o3d.core.Dtype.UInt32)

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(vertices, triangles)

    rs = np.random.RandomState(123)
    rays = o3d.core.Tensor.from_numpy(
        rs.uniform(size=shape + [6]).astype(np.float32))
    query_points = o3d.core.Tensor.from_numpy(
        rs.uniform(size=shape + [3]).astype(np.float32))

    ans = scene.count_intersections(rays)
    assert list(ans.shape) == shape

    ans = scene.compute_distance(query_points)
    assert list(ans.shape) == shape

    ans = scene.compute_signed_distance(query_points)
    assert list(ans.shape) == shape

    ans = scene.compute_occupancy(query_points)
    assert list(ans.shape) == shape

    # some outputs append a specific last dim
    last_dim = {
        't_hit': [],
        'geometry_ids': [],
        'primitive_ids': [],
        'primitive_uvs': [2],
        'primitive_normals': [3],
        'points': [3]
    }

    ans = scene.cast_rays(rays)
    for k, v in ans.items():
        expected_shape = shape + last_dim[k]
        assert list(
            v.shape
        ) == expected_shape, 'shape mismatch: expected {} but got {} for {}'.format(
            expected_shape, list(v.shape), k)

    ans = scene.compute_closest_points(query_points)
    for k, v in ans.items():
        expected_shape = shape + last_dim[k]
        assert list(
            v.shape
        ) == expected_shape, 'shape mismatch: expected {} but got {} for {}'.format(
            expected_shape, list(v.shape), k)
