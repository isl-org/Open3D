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
import open3d.core as o3c
import numpy as np
import pytest
import pickle
import tempfile

import sys
import os

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../..")
from open3d_test import list_devices


def test_clip_plane():
    cube = o3d.t.geometry.TriangleMesh.from_legacy(
        o3d.geometry.TriangleMesh.create_box())
    clipped_cube = cube.clip_plane(point=[0.5, 0, 0], normal=[1, 0, 0])
    assert clipped_cube.vertex.positions.shape == (12, 3)
    assert clipped_cube.triangle.indices.shape == (14, 3)


def test_slice_plane():
    box = o3d.t.geometry.TriangleMesh.create_box()
    slices = box.slice_plane([0, 0.5, 0], [1, 1, 1], [-0.1, 0, 0.1])
    assert slices.point.positions.shape == (9, 3)
    assert slices.line.indices.shape == (9, 2)


@pytest.mark.parametrize("device", list_devices())
def test_create_box(device):
    # Test with default parameters.
    box_default = o3d.t.geometry.TriangleMesh.create_box(device=device)

    vertex_positions_default = o3c.Tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 1.0],
         [0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
        o3c.float32, device)

    triangle_indices_default = o3c.Tensor(
        [[4, 7, 5], [4, 6, 7], [0, 2, 4], [2, 6, 4], [0, 1, 2], [1, 3, 2],
         [1, 5, 7], [1, 7, 3], [2, 3, 7], [2, 7, 6], [0, 4, 1], [1, 4, 5]],
        o3c.int64, device)

    assert box_default.vertex.positions.allclose(vertex_positions_default)
    assert box_default.triangle.indices.allclose(triangle_indices_default)

    # Test with custom parameters.
    box_custom = o3d.t.geometry.TriangleMesh.create_box(2, 3, 4, o3c.float64,
                                                        o3c.int32, device)

    vertex_positions_custom = o3c.Tensor(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 4.0], [2.0, 0.0, 4.0],
         [0.0, 3.0, 0.0], [2.0, 3.0, 0.0], [0.0, 3.0, 4.0], [2.0, 3.0, 4.0]],
        o3c.float64, device)

    triangle_indices_custom = o3c.Tensor(
        [[4, 7, 5], [4, 6, 7], [0, 2, 4], [2, 6, 4], [0, 1, 2], [1, 3, 2],
         [1, 5, 7], [1, 7, 3], [2, 3, 7], [2, 7, 6], [0, 4, 1], [1, 4, 5]],
        o3c.int32, device)

    assert box_custom.vertex.positions.allclose(vertex_positions_custom)
    assert box_custom.triangle.indices.allclose(triangle_indices_custom)


@pytest.mark.parametrize("device", list_devices())
def test_create_sphere(device):
    # Test with custom parameters.
    sphere_custom = o3d.t.geometry.TriangleMesh.create_sphere(
        1, 3, o3c.float64, o3c.int32, device)

    vertex_positions_custom = o3c.Tensor(
        [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0], [0.866025, 0, 0.5],
         [0.433013, 0.75, 0.5], [-0.433013, 0.75, 0.5], [-0.866025, 0.0, 0.5],
         [-0.433013, -0.75, 0.5], [0.433013, -0.75, 0.5], [0.866025, 0.0, -0.5],
         [0.433013, 0.75, -0.5], [-0.433013, 0.75, -0.5],
         [-0.866025, 0.0, -0.5], [-0.433013, -0.75, -0.5],
         [0.433013, -0.75, -0.5]], o3c.float64, device)

    triangle_indices_custom = o3c.Tensor(
        [[0, 2, 3], [1, 9, 8], [0, 3, 4], [1, 10, 9], [0, 4, 5], [1, 11, 10],
         [0, 5, 6], [1, 12, 11], [0, 6, 7], [1, 13, 12], [0, 7, 2], [1, 8, 13],
         [8, 3, 2], [8, 9, 3], [9, 4, 3], [9, 10, 4], [10, 5, 4], [10, 11, 5],
         [11, 6, 5], [11, 12, 6], [12, 7, 6], [12, 13, 7], [13, 2, 7],
         [13, 8, 2]], o3c.int32, device)

    assert sphere_custom.vertex.positions.allclose(vertex_positions_custom)
    assert sphere_custom.triangle.indices.allclose(triangle_indices_custom)


@pytest.mark.parametrize("device", list_devices())
def test_create_tetrahedron(device):
    # Test with custom parameters.
    tetrahedron_custom = o3d.t.geometry.TriangleMesh.create_tetrahedron(
        2, o3c.float64, o3c.int32, device)

    vertex_positions_custom = o3c.Tensor(
        [[1.88562, 0.0, -0.666667], [-0.942809, 1.63299, -0.666667],
         [-0.942809, -1.63299, -0.666667], [0.0, 0.0, 2]], o3c.float64, device)

    triangle_indices_custom = o3c.Tensor(
        [[0, 2, 1], [0, 3, 2], [0, 1, 3], [1, 2, 3]], o3c.int32, device)

    assert tetrahedron_custom.vertex.positions.allclose(vertex_positions_custom)
    assert tetrahedron_custom.triangle.indices.allclose(triangle_indices_custom)


@pytest.mark.parametrize("device", list_devices())
def test_create_octahedron(device):
    # Test with custom parameters.
    octahedron_custom = o3d.t.geometry.TriangleMesh.create_octahedron(
        2, o3c.float64, o3c.int32, device)

    vertex_positions_custom = o3c.Tensor(
        [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0], [-2.0, 0.0, 0.0],
         [0.0, -2.0, 0.0], [0.0, 0.0, -2.0]], o3c.float64, device)

    triangle_indices_custom = o3c.Tensor(
        [[0, 1, 2], [1, 3, 2], [3, 4, 2], [4, 0, 2], [0, 5, 1], [1, 5, 3],
         [3, 5, 4], [4, 5, 0]], o3c.int32, device)

    assert octahedron_custom.vertex.positions.allclose(vertex_positions_custom)
    assert octahedron_custom.triangle.indices.allclose(triangle_indices_custom)


@pytest.mark.parametrize("device", list_devices())
def test_create_icosahedron(device):
    # Test with custom parameters.
    icosahedron_custom = o3d.t.geometry.TriangleMesh.create_icosahedron(
        2, o3c.float64, o3c.int32, device)

    vertex_positions_custom = o3c.Tensor(
        [[-2.0, 0.0, 3.23607], [2.0, 0.0, 3.23607], [2.0, 0.0, -3.23607],
         [-2.0, 0.0, -3.23607], [0.0, -3.23607, 2.0], [0.0, 3.23607, 2.0],
         [0.0, 3.23607, -2.0], [0.0, -3.23607, -2.0], [-3.23607, -2.0, 0.0],
         [3.23607, -2.0, 0.0], [3.23607, 2.0, 0.0], [-3.23607, 2.0, 0.0]],
        o3c.float64, device)

    triangle_indices_custom = o3c.Tensor(
        [[0, 4, 1], [0, 1, 5], [1, 4, 9], [1, 9, 10], [1, 10, 5], [0, 8, 4],
         [0, 11, 8], [0, 5, 11], [5, 6, 11], [5, 10, 6], [4, 8, 7], [4, 7, 9],
         [3, 6, 2], [3, 2, 7], [2, 6, 10], [2, 10, 9], [2, 9, 7], [3, 11, 6],
         [3, 8, 11], [3, 7, 8]], o3c.int32, device)

    assert icosahedron_custom.vertex.positions.allclose(vertex_positions_custom)
    assert icosahedron_custom.triangle.indices.allclose(triangle_indices_custom)


@pytest.mark.parametrize("device", list_devices())
def test_create_cylinder(device):
    # Test with custom parameters.
    cylinder_custom = o3d.t.geometry.TriangleMesh.create_cylinder(
        1, 2, 3, 3, o3c.float64, o3c.int32, device)

    vertex_positions_custom = o3c.Tensor(
        [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0], [1.0, 0.0, 1.0],
         [-0.5, 0.866025, 1.0], [-0.5, -0.866025, 1.0], [1.0, 0.0, 0.333333],
         [-0.5, 0.866025, 0.333333], [-0.5, -0.866025, 0.333333],
         [1.0, 0.0, -0.333333], [-0.5, 0.866025, -0.333333],
         [-0.5, -0.866025, -0.333333], [1.0, 0.0, -1.0], [-0.5, 0.866025, -1.0],
         [-0.5, -0.866025, -1.0]], o3c.float64, device)

    triangle_indices_custom = o3c.Tensor(
        [[0, 2, 3], [1, 12, 11], [0, 3, 4], [1, 13, 12], [0, 4, 2], [1, 11, 13],
         [5, 3, 2], [5, 6, 3], [6, 4, 3], [6, 7, 4], [7, 2, 4], [7, 5, 2],
         [8, 6, 5], [8, 9, 6], [9, 7, 6], [9, 10, 7], [10, 5, 7], [10, 8, 5],
         [11, 9, 8], [11, 12, 9], [12, 10, 9], [12, 13, 10], [13, 8, 10],
         [13, 11, 8]], o3c.int32, device)

    assert cylinder_custom.vertex.positions.allclose(vertex_positions_custom)
    assert cylinder_custom.triangle.indices.allclose(triangle_indices_custom)


@pytest.mark.parametrize("device", list_devices())
def test_create_cone(device):
    # Test with custom parameters.
    cone_custom = o3d.t.geometry.TriangleMesh.create_cone(
        2, 4, 3, 2, o3c.float64, o3c.int32, device)

    vertex_positions_custom = o3c.Tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 4.0], [2.0, 0.0, 0.0],
         [-1.0, 1.73205, 0.0], [-1.0, -1.73205, 0.0], [1.0, 0.0, 2.0],
         [-0.5, 0.866025, 2], [-0.5, -0.866025, 2]], o3c.float64, device)

    triangle_indices_custom = o3c.Tensor(
        [[0, 3, 2], [1, 5, 6], [0, 4, 3], [1, 6, 7], [0, 2, 4], [1, 7, 5],
         [6, 2, 3], [6, 5, 2], [7, 3, 4], [7, 6, 3], [5, 4, 2], [5, 7, 4]],
        o3c.int32, device)

    assert cone_custom.vertex.positions.allclose(vertex_positions_custom)
    assert cone_custom.triangle.indices.allclose(triangle_indices_custom)


@pytest.mark.parametrize("device", list_devices())
def test_create_torus(device):
    # Test with custom parameters.
    torus_custom = o3d.t.geometry.TriangleMesh.create_torus(
        2, 1, 6, 3, o3c.float64, o3c.int32, device)

    vertex_positions_custom = o3c.Tensor(
        [[3.0, 0.0, 0.0], [1.5, 0.0, 0.866025], [1.5, 0.0, -0.866025],
         [1.5, 2.59808, 0.0], [0.75, 1.29904, 0.866025],
         [0.75, 1.29904, -0.866025], [-1.5, 2.59808, 0],
         [-0.75, 1.29904, 0.866025], [-0.75, 1.29904, -0.866025],
         [-3.0, 0.0, 0.0], [-1.5, 0.0, 0.866025], [-1.5, 0.0, -0.866025],
         [-1.5, -2.59808, 0.0], [-0.75, -1.29904, 0.866025],
         [-0.75, -1.29904, -0.866025], [1.5, -2.59808, 0.0],
         [0.75, -1.29904, 0.866025], [0.75, -1.29904, -0.866025]], o3c.float64,
        device)

    triangle_indices_custom = o3c.Tensor(
        [[3, 4, 0], [0, 4, 1], [4, 5, 1], [1, 5, 2], [5, 3, 2], [2, 3, 0],
         [6, 7, 3], [3, 7, 4], [7, 8, 4], [4, 8, 5], [8, 6, 5], [5, 6, 3],
         [9, 10, 6], [6, 10, 7], [10, 11, 7], [7, 11, 8], [11, 9, 8], [8, 9, 6],
         [12, 13, 9], [9, 13, 10], [13, 14, 10], [10, 14, 11], [14, 12, 11],
         [11, 12, 9], [15, 16, 12], [12, 16, 13], [16, 17, 13], [13, 17, 14],
         [17, 15, 14], [14, 15, 12], [0, 1, 15], [15, 1, 16], [1, 2, 16],
         [16, 2, 17], [2, 0, 17], [17, 0, 15]], o3c.int32, device)

    assert torus_custom.vertex.positions.allclose(vertex_positions_custom)
    assert torus_custom.triangle.indices.allclose(triangle_indices_custom)


@pytest.mark.parametrize("device", list_devices())
def test_create_arrow(device):
    # Test with custom parameters.
    arrow_custom = o3d.t.geometry.TriangleMesh.create_arrow(
        1, 2, 4, 2, 4, 1, 1, o3c.float64, o3c.int32, device)

    vertex_positions_custom = o3c.Tensor(
        [[0.0, 0.0, 4.0], [0.0, 0.0, 0.0], [1.0, 0.0, 4.0], [0.0, 1.0, 4.0],
         [-1.0, 0.0, 4.0], [0.0, -1.0, 4.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
         [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 4.0], [0.0, 0.0, 6.0],
         [2.0, 0.0, 4.0], [0.0, 2.0, 4.0], [-2.0, 0.0, 4.0], [0.0, -2.0, 4.0]],
        o3c.float64, device)

    triangle_indices_custom = o3c.Tensor(
        [[0, 2, 3], [1, 7, 6], [0, 3, 4], [1, 8, 7], [0, 4, 5], [1, 9, 8],
         [0, 5, 2], [1, 6, 9], [6, 3, 2], [6, 7, 3], [7, 4, 3], [7, 8, 4],
         [8, 5, 4], [8, 9, 5], [9, 2, 5], [9, 6, 2], [10, 13, 12], [11, 12, 13],
         [10, 14, 13], [11, 13, 14], [10, 15, 14], [11, 14, 15], [10, 12, 15],
         [11, 15, 12]], o3c.int32, device)

    assert arrow_custom.vertex.positions.allclose(vertex_positions_custom)
    assert arrow_custom.triangle.indices.allclose(triangle_indices_custom)


@pytest.mark.parametrize("device", list_devices())
def test_create_mobius(device):
    # Test with custom parameters.
    mobius_custom = o3d.t.geometry.TriangleMesh.create_mobius(
        10, 2, 1, 1, 1, 1, 1, o3c.float64, o3c.int32, device)

    vertex_positions_custom = o3c.Tensor(
        [[0.5, 0.0, 0.0], [1.5, 0.0, 0.0], [0.424307, 0.308277, -0.154508],
         [1.19373, 0.867294, 0.154508], [0.184017, 0.566346, -0.293893],
         [0.434017, 1.33577, 0.293893], [-0.218199, 0.671548, -0.404508],
         [-0.399835, 1.23057, 0.404508], [-0.684017, 0.496967, -0.475528],
         [-0.934017, 0.678603, 0.475528], [-1.0, 0.0, -0.5], [-1.0, 0.0, 0.5],
         [-0.934017, -0.678603, -0.475528], [-0.684017, -0.496967, 0.475528],
         [-0.399835, -1.23057, -0.404508], [-0.218199, -0.671548, 0.404508],
         [0.434017, -1.33577, -0.293893], [0.184017, -0.566346, 0.293893],
         [1.19373, -0.867294, -0.154508], [0.424307, -0.308277, 0.154508]],
        o3c.float64, device)

    triangle_indices_custom = o3c.Tensor(
        [[0, 3, 1], [0, 2, 3], [3, 2, 4], [3, 4, 5], [4, 7, 5], [4, 6, 7],
         [7, 6, 8], [7, 8, 9], [8, 11, 9], [8, 10, 11], [11, 10, 12],
         [11, 12, 13], [12, 15, 13], [12, 14, 15], [15, 14, 16], [15, 16, 17],
         [16, 19, 17], [16, 18, 19], [18, 19, 1], [1, 19, 0]], o3c.int32,
        device)

    assert mobius_custom.vertex.positions.allclose(vertex_positions_custom)
    assert mobius_custom.triangle.indices.allclose(triangle_indices_custom)


def test_create_text():
    mesh = o3d.t.geometry.TriangleMesh.create_text('Open3D', depth=1)
    assert mesh.vertex.positions.shape == (624, 3)
    assert mesh.triangle.indices.shape == (936, 3)


def test_simplify_quadric_decimation():
    cube = o3d.t.geometry.TriangleMesh.from_legacy(
        o3d.geometry.TriangleMesh.create_box().subdivide_midpoint(3))

    # Chose reduction factor such that we get 12 faces.
    target_reduction = 1 - (12 / cube.triangle.indices.shape[0])
    simplified = cube.simplify_quadric_decimation(
        target_reduction=target_reduction)

    assert simplified.vertex.positions.shape == (8, 3)
    assert simplified.triangle.indices.shape == (12, 3)


def test_boolean_operations():
    box = o3d.geometry.TriangleMesh.create_box()
    box = o3d.t.geometry.TriangleMesh.from_legacy(box)
    sphere = o3d.geometry.TriangleMesh.create_sphere(0.8)
    sphere = o3d.t.geometry.TriangleMesh.from_legacy(sphere)
    # check input sphere
    assert sphere.vertex.positions.shape == (762, 3)
    assert sphere.triangle.indices.shape == (1520, 3)

    ans = box.boolean_union(sphere)
    assert ans.vertex.positions.shape == (730, 3)
    assert ans.triangle.indices.shape == (1384, 3)

    ans = box.boolean_intersection(sphere)
    assert ans.vertex.positions.shape == (154, 3)
    assert ans.triangle.indices.shape == (232, 3)

    ans = box.boolean_difference(sphere)
    assert ans.vertex.positions.shape == (160, 3)
    assert ans.triangle.indices.shape == (244, 3)


def test_hole_filling():
    sphere = o3d.geometry.TriangleMesh.create_sphere(1.0)
    sphere = o3d.t.geometry.TriangleMesh.from_legacy(sphere)
    clipped = sphere.clip_plane([0.8, 0, 0], [1, 0, 0])
    assert not clipped.to_legacy().is_watertight()
    filled = clipped.fill_holes()
    assert filled.to_legacy().is_watertight()


def test_uvatlas():
    box = o3d.t.geometry.TriangleMesh.create_box()
    box.compute_uvatlas()
    assert box.triangle['texture_uvs'].shape == (12, 3, 2)


def test_bake_vertex_attr_textures():
    desired = np.array([
        [[0., 0., 0.], [0., 0., 0.], [1., 0.25, 0.75], [1., 0.75, 0.75],
         [0., 0., 0.], [0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
        [[0., 0., 0.], [0., 0., 0.], [1., 0.25, 0.25], [1., 0.75, 0.25],
         [0., 0., 0.], [0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
        [[0.75, 0., 0.75], [0.75, 0., 0.25], [0.75, 0.25, 0.], [0.75, 0.75, 0.],
         [0.75, 1., 0.25], [0.75, 1., 0.75], [0., 0., 0.], [0., 0., 0.]],
        [[0.25, 0., 0.75], [0.25, 0., 0.25], [0.25, 0.25, 0.], [0.25, 0.75, 0.],
         [0.25, 1., 0.25], [0.25, 1., 0.75], [0., 0., 0.], [0., 0., 0.]],
        [[0., 0., 0.], [0., 0., 0.], [0., 0.25, 0.25], [0., 0.75, 0.25],
         [0., 0., 0.], [0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
        [[0., 0., 0.], [0., 0., 0.], [0., 0.25, 0.75], [0., 0.75, 0.75],
         [0., 0., 0.], [0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
        [[0., 0., 0.], [0., 0., 0.], [0.25, 0.25, 1.], [0.25, 0.75, 1.],
         [0., 0., 0.], [0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
        [[0., 0., 0.], [0., 0., 0.], [0.75, 0.25, 1.], [0.75, 0.75, 1.],
         [0., 0., 0.], [0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]
    ],
                       dtype=np.float32)

    box = o3d.geometry.TriangleMesh.create_box(create_uv_map=True)
    box = o3d.t.geometry.TriangleMesh.from_legacy(box)
    textures = box.bake_vertex_attr_textures(8, {'positions'}, margin=0.1)

    np.testing.assert_allclose(textures['positions'].numpy(), desired)


def test_bake_triangle_attr_textures():
    desired = np.array(
        [[-1, -1, 7, 7, -1, -1, -1, -1], [-1, -1, 7, 6, -1, -1, -1, -1],
         [5, 5, 10, 11, 0, 0, -1, -1], [5, 4, 10, 10, 0, 1, -1, -1],
         [-1, -1, 2, 2, -1, -1, -1, -1], [-1, -1, 2, 3, -1, -1, -1, -1],
         [-1, -1, 8, 9, -1, -1, -1, -1], [-1, -1, 8, 8, -1, -1, -1, -1]],
        dtype=np.int64)

    box = o3d.geometry.TriangleMesh.create_box(create_uv_map=True)
    box = o3d.t.geometry.TriangleMesh.from_legacy(box)
    box.triangle.index = np.arange(box.triangle.indices.shape[0])
    # shift the uvs to avoid pixel centers exactly at triangle boundaries.
    box.triangle.texture_uvs[:, :, 0] += 0.01

    textures = box.bake_triangle_attr_textures(8, {'index'},
                                               margin=0.1,
                                               fill=-1)
    np.testing.assert_equal(textures['index'].numpy(), desired)


def test_extrude_rotation():
    mesh = o3d.t.geometry.TriangleMesh([[1, 1, 0], [0.7, 1, 0], [1, 0.7, 0]],
                                       [[0, 1, 2]])
    ans = mesh.extrude_rotation(3 * 360, [0, 1, 0],
                                resolution=3 * 16,
                                translation=2)
    assert ans.vertex.positions.shape == (147, 3)
    assert ans.triangle.indices.shape == (290, 3)


def test_extrude_linear():
    triangle = o3d.t.geometry.TriangleMesh(
        [[1.0, 1.0, 0.0], [0, 1, 0], [1, 0, 0]], [[0, 1, 2]])
    ans = triangle.extrude_linear([0, 0, 1])
    assert ans.vertex.positions.shape == (6, 3)
    assert ans.triangle.indices.shape == (8, 3)


@pytest.mark.parametrize("device", list_devices())
def test_pickle(device):
    mesh = o3d.t.geometry.TriangleMesh.create_box().to(device)
    with tempfile.TemporaryDirectory() as temp_dir:
        file_name = f"{temp_dir}/mesh.pkl"
        pickle.dump(mesh, open(file_name, "wb"))
        mesh_load = pickle.load(open(file_name, "rb"))
        assert mesh_load.device == device
        assert mesh_load.vertex.positions.dtype == o3c.float32
        assert mesh_load.triangle.indices.dtype == o3c.int64
        np.testing.assert_equal(mesh_load.vertex.positions.cpu().numpy(),
                                mesh.vertex.positions.cpu().numpy())
        np.testing.assert_equal(mesh_load.triangle.indices.cpu().numpy(),
                                mesh.triangle.indices.cpu().numpy())
