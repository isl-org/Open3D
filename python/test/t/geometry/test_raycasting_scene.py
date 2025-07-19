# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import open3d as o3d
import numpy as np
import pytest

import sys
import os

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../..")
from open3d_test import list_devices


# test intersection with a single triangle
@pytest.mark.parametrize("device",
                         list_devices(enable_cuda=False, enable_sycl=True))
def test_cast_rays(device):
    vertices = o3d.core.Tensor([[0, 0, 0], [1, 0, 0], [1, 1, 0]],
                               dtype=o3d.core.float32,
                               device=device)
    triangles = o3d.core.Tensor([[0, 1, 2]],
                                dtype=o3d.core.uint32,
                                device=device)

    scene = o3d.t.geometry.RaycastingScene(device=device)
    geom_id = scene.add_triangles(vertices, triangles)

    rays = o3d.core.Tensor(
        [[0.2, 0.1, 1, 0, 0, -1], [10, 10, 10, 1, 0, 0]],
        dtype=o3d.core.float32,
        device=device,
    )
    ans = scene.cast_rays(rays)

    # first ray hits the triangle
    assert geom_id == ans["geometry_ids"][0].cpu()
    assert np.isclose(ans["t_hit"][0].item(), 1.0)

    # second ray misses
    assert o3d.t.geometry.RaycastingScene.INVALID_ID == ans["geometry_ids"][
        1].cpu()
    assert np.isinf(ans["t_hit"][1].item())


# cast lots of random rays to test the internal batching
# we expect no errors for this test
@pytest.mark.parametrize("device",
                         list_devices(enable_cuda=False, enable_sycl=True))
def test_cast_lots_of_rays(device):
    vertices = o3d.core.Tensor([[0, 0, 0], [1, 0, 0], [1, 1, 0]],
                               dtype=o3d.core.float32,
                               device=device)
    triangles = o3d.core.Tensor([[0, 1, 2]],
                                dtype=o3d.core.uint32,
                                device=device)

    scene = o3d.t.geometry.RaycastingScene(device=device)
    scene.add_triangles(vertices, triangles)

    rs = np.random.RandomState(123)
    rays = o3d.core.Tensor.from_numpy(rs.rand(7654321, 6).astype(np.float32))
    rays = rays.to(device)

    _ = scene.cast_rays(rays)


# test occlusion with a single triangle
@pytest.mark.parametrize("device",
                         list_devices(enable_cuda=False, enable_sycl=True))
def test_test_occlusions(device):
    vertices = o3d.core.Tensor([[0, 0, 0], [1, 0, 0], [1, 1, 0]],
                               dtype=o3d.core.float32,
                               device=device)
    triangles = o3d.core.Tensor([[0, 1, 2]],
                                dtype=o3d.core.uint32,
                                device=device)

    scene = o3d.t.geometry.RaycastingScene(device=device)
    scene.add_triangles(vertices, triangles)

    rays = o3d.core.Tensor(
        [[0.2, 0.1, 1, 0, 0, -1], [10, 10, 10, 1, 0, 0]],
        dtype=o3d.core.float32,
        device=device,
    )
    ans = scene.test_occlusions(rays).cpu()

    # first ray is occluded by the triangle
    assert ans[0] == True

    # second ray is not occluded
    assert ans[1] == False

    # set tfar such that no ray is occluded
    ans = scene.test_occlusions(rays, tfar=0.5).cpu()
    assert ans.any() == False

    # set tnear such that no ray is occluded
    ans = scene.test_occlusions(rays, tnear=1.5).cpu()
    assert ans.any() == False


# test lots of random rays for occlusions to test the internal batching
# we expect no errors for this test
@pytest.mark.parametrize("device",
                         list_devices(enable_cuda=False, enable_sycl=True))
def test_test_lots_of_occlusions(device):
    vertices = o3d.core.Tensor([[0, 0, 0], [1, 0, 0], [1, 1, 0]],
                               dtype=o3d.core.float32,
                               device=device)
    triangles = o3d.core.Tensor([[0, 1, 2]],
                                dtype=o3d.core.uint32,
                                device=device)

    scene = o3d.t.geometry.RaycastingScene(device=device)
    scene.add_triangles(vertices, triangles)

    rs = np.random.RandomState(123)
    rays = o3d.core.Tensor.from_numpy(rs.rand(7654321, 6).astype(np.float32))
    rays = rays.to(device)

    _ = scene.test_occlusions(rays)


@pytest.mark.parametrize("device",
                         list_devices(enable_cuda=False, enable_sycl=True))
def test_add_triangle_mesh(device):
    cube = o3d.t.geometry.TriangleMesh.create_box()
    cube = cube.to(device)

    scene = o3d.t.geometry.RaycastingScene(device=device)
    scene.add_triangles(cube)

    rays = o3d.core.Tensor(
        [[0.5, 0.5, -1, 0, 0, 1], [0.5, 0.5, 0.5, 0, 0, 1],
         [10, 10, 10, 1, 0, 0]],
        dtype=o3d.core.float32,
        device=device,
    )
    ans = scene.count_intersections(rays)

    np.testing.assert_equal(ans.cpu().numpy(), [2, 1, 0])


@pytest.mark.parametrize("device",
                         list_devices(enable_cuda=False, enable_sycl=True))
def test_count_intersections(device):
    cube = o3d.t.geometry.TriangleMesh.create_box()
    vertex_positions = cube.vertex.positions
    vertex_positions = vertex_positions.to(device)
    triangle_indices = cube.triangle.indices
    triangle_indices = triangle_indices.to(o3d.core.Dtype.UInt32)
    triangle_indices = triangle_indices.to(device)

    scene = o3d.t.geometry.RaycastingScene(device=device)
    scene.add_triangles(vertex_positions, triangle_indices)

    rays = o3d.core.Tensor(
        [[0.5, 0.5, -1, 0, 0, 1], [0.5, 0.5, 0.5, 0, 0, 1],
         [10, 10, 10, 1, 0, 0]],
        dtype=o3d.core.float32,
        device=device,
    )
    ans = scene.count_intersections(rays)

    np.testing.assert_equal(ans.cpu().numpy(), [2, 1, 0])


# count lots of random ray intersections to test the internal batching
# we expect no errors for this test
@pytest.mark.parametrize("device",
                         list_devices(enable_cuda=False, enable_sycl=True))
def test_count_lots_of_intersections(device):
    cube = o3d.t.geometry.TriangleMesh.create_box()
    vertex_positions = cube.vertex.positions
    vertex_positions = vertex_positions.to(device)
    triangle_indices = cube.triangle.indices
    triangle_indices = triangle_indices.to(o3d.core.Dtype.UInt32)
    triangle_indices = triangle_indices.to(device)

    scene = o3d.t.geometry.RaycastingScene(device=device)
    scene.add_triangles(vertex_positions, triangle_indices)

    rs = np.random.RandomState(123)
    rays = o3d.core.Tensor.from_numpy(rs.rand(1234567, 6).astype(np.float32))
    rays = rays.to(device)

    _ = scene.count_intersections(rays)


@pytest.mark.parametrize("device",
                         list_devices(enable_cuda=False, enable_sycl=True))
def test_list_intersections(device):
    cube = o3d.t.geometry.TriangleMesh.create_box()
    vertex_positions = cube.vertex.positions
    vertex_positions = vertex_positions.to(device)
    triangle_indices = cube.triangle.indices
    triangle_indices = triangle_indices.to(o3d.core.Dtype.UInt32)
    triangle_indices = triangle_indices.to(device)

    scene = o3d.t.geometry.RaycastingScene(device=device)
    scene.add_triangles(vertex_positions, triangle_indices)

    rays = o3d.core.Tensor(
        [[0.5, 0.5, -1, 0, 0, 1], [0.5, 0.5, 0.5, 0, 0, 1],
         [10, 10, 10, 1, 0, 0]],
        dtype=o3d.core.float32,
        device=device,
    )
    ans = scene.list_intersections(rays)

    np.testing.assert_allclose(ans["t_hit"].cpu().numpy(),
                               np.array([1.0, 2.0, 0.5]),
                               rtol=1e-6,
                               atol=1e-6)


# list lots of random ray intersections to test the internal batching
# we expect no errors for this test
@pytest.mark.parametrize("device",
                         list_devices(enable_cuda=False, enable_sycl=True))
def test_list_lots_of_intersections(device):
    cube = o3d.t.geometry.TriangleMesh.create_box()
    vertex_positions = cube.vertex.positions
    vertex_positions = vertex_positions.to(device)
    triangle_indices = cube.triangle.indices
    triangle_indices = triangle_indices.to(o3d.core.Dtype.UInt32)
    triangle_indices = triangle_indices.to(device)

    scene = o3d.t.geometry.RaycastingScene(device=device)
    scene.add_triangles(vertex_positions, triangle_indices)

    rs = np.random.RandomState(123)
    rays = o3d.core.Tensor.from_numpy(rs.rand(123456, 6).astype(np.float32))
    rays = rays.to(device)

    _ = scene.list_intersections(rays)


def test_compute_closest_points():
    vertices = o3d.core.Tensor([[0, 0, 0], [1, 0, 0], [1, 1, 0]],
                               dtype=o3d.core.float32)
    triangles = o3d.core.Tensor([[0, 1, 2]], dtype=o3d.core.uint32)

    scene = o3d.t.geometry.RaycastingScene()
    geom_id = scene.add_triangles(vertices, triangles)

    query_points = o3d.core.Tensor([[0.2, 0.1, 1], [10, 10, 10]],
                                   dtype=o3d.core.float32)
    ans = scene.compute_closest_points(query_points)

    assert (geom_id == ans["geometry_ids"]).all()
    assert (0 == ans["primitive_ids"]).all()
    np.testing.assert_allclose(
        ans["points"].numpy(),
        np.array([[0.2, 0.1, 0.0], [1, 1, 0]]),
        rtol=1e-6,
        atol=1e-6,
    )


# compute lots of closest points to test the internal batching
# we expect no errors for this test
def test_compute_lots_of_closest_points():
    vertices = o3d.core.Tensor([[0, 0, 0], [1, 0, 0], [1, 1, 0]],
                               dtype=o3d.core.float32)
    triangles = o3d.core.Tensor([[0, 1, 2]], dtype=o3d.core.uint32)

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(vertices, triangles)

    rs = np.random.RandomState(123)
    query_points = o3d.core.Tensor.from_numpy(
        rs.rand(1234567, 3).astype(np.float32))
    _ = scene.compute_closest_points(query_points)


def test_compute_distance():
    cube = o3d.t.geometry.TriangleMesh.create_box()

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(cube)

    query_points = o3d.core.Tensor(
        [[0.5, 0.5, 0.5], [-0.5, -0.5, -0.5], [0, 0, 0]],
        dtype=o3d.core.float32)
    ans = scene.compute_distance(query_points)
    np.testing.assert_allclose(ans.numpy(), [0.5, np.sqrt(3 * 0.5**2), 0.0])


def test_compute_signed_distance():
    cube = o3d.t.geometry.TriangleMesh.create_box()

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(cube)

    query_points = o3d.core.Tensor(
        [[0.5, 0.5, 0.5], [-0.5, -0.5, -0.5], [0, 0, 0]],
        dtype=o3d.core.float32)
    ans = scene.compute_signed_distance(query_points)
    np.testing.assert_allclose(ans.numpy(), [-0.5, np.sqrt(3 * 0.5**2), 0.0])


def test_compute_occupancy():
    cube = o3d.t.geometry.TriangleMesh.create_box()

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(cube)

    query_points = o3d.core.Tensor([[0.5, 0.5, 0.5], [-0.5, -0.5, -0.5]],
                                   dtype=o3d.core.float32)
    ans = scene.compute_occupancy(query_points)
    np.testing.assert_allclose(ans.numpy(), [1.0, 0.0])


@pytest.mark.parametrize("shape", ([11], [1, 2, 3], [32, 14]))
def test_output_shapes(shape):
    vertices = o3d.core.Tensor([[0, 0, 0], [1, 0, 0], [1, 1, 0]],
                               dtype=o3d.core.float32)
    triangles = o3d.core.Tensor([[0, 1, 2]], dtype=o3d.core.uint32)

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
        "t_hit": [],
        "geometry_ids": [],
        "primitive_ids": [],
        "primitive_uvs": [2],
        "primitive_normals": [3],
        "points": [3],
        "ray_ids": [],
        "ray_splits": [],
    }

    ans = scene.cast_rays(rays)
    for k, v in ans.items():
        expected_shape = shape + last_dim[k]
        assert (list(v.shape) == expected_shape
               ), "shape mismatch: expected {} but got {} for {}".format(
                   expected_shape, list(v.shape), k)

    ans = scene.compute_closest_points(query_points)
    for k, v in ans.items():
        expected_shape = shape + last_dim[k]
        assert (list(v.shape) == expected_shape
               ), "shape mismatch: expected {} but got {} for {}".format(
                   expected_shape, list(v.shape), k)

    ans = scene.list_intersections(rays)
    nx = np.sum(scene.count_intersections(rays).numpy()).tolist()
    for k, v in ans.items():
        if k == "ray_splits":
            alt_shape = [np.prod(rays.shape[:-1]) + 1]
        else:
            alt_shape = [nx]
        # use np.append otherwise issues if alt_shape = [0] and last_dim[k] = []
        expected_shape = np.append(alt_shape, last_dim[k]).tolist()
        assert (list(v.shape) == expected_shape
               ), "shape mismatch: expected {} but got {} for {}".format(
                   expected_shape, list(v.shape), k)


def test_sphere_wrong_occupancy():
    # This test checks a specific scenario where the old implementation
    # without ray jitter produced wrong results for a sphere because some
    # rays miss hitting exactly a vertex or an edge.
    mesh = o3d.t.geometry.TriangleMesh.create_sphere(0.8)

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh)

    min_bound = mesh.vertex.positions.min(0).numpy() * 1.1
    max_bound = mesh.vertex.positions.max(0).numpy() * 1.1

    xyz_range = np.linspace(min_bound, max_bound, num=6)
    query_points = np.stack(np.meshgrid(*xyz_range.T),
                            axis=-1).astype(np.float32)

    occupancy = scene.compute_occupancy(query_points)
    expected = np.array(
        [
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
        ],
        dtype=np.float32,
    )
    np.testing.assert_equal(occupancy.numpy(), expected)

    # we should get the same result with more samples
    occupancy_3samples = scene.compute_occupancy(query_points, nsamples=3)
    np.testing.assert_equal(occupancy_3samples.numpy(), expected)
