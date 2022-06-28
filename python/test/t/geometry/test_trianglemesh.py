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


def test_clip_plane():
    cube = o3d.t.geometry.TriangleMesh.from_legacy(
        o3d.geometry.TriangleMesh.create_box())
    clipped_cube = cube.clip_plane(point=[0.5, 0, 0], normal=[1, 0, 0])
    assert clipped_cube.vertex['positions'].shape == (12, 3)
    assert clipped_cube.triangle['indices'].shape == (14, 3)


def test_simplify_quadric_decimation():
    cube = o3d.t.geometry.TriangleMesh.from_legacy(
        o3d.geometry.TriangleMesh.create_box().subdivide_midpoint(3))

    # chose reduction factor such that we get 12 faces
    target_reduction = 1 - (12 / cube.triangle['indices'].shape[0])
    simplified = cube.simplify_quadric_decimation(
        target_reduction=target_reduction)

    assert simplified.vertex['positions'].shape == (8, 3)
    assert simplified.triangle['indices'].shape == (12, 3)


def test_boolean_operations():
    box = o3d.geometry.TriangleMesh.create_box()
    box = o3d.t.geometry.TriangleMesh.from_legacy(box)
    sphere = o3d.geometry.TriangleMesh.create_sphere(0.8)
    sphere = o3d.t.geometry.TriangleMesh.from_legacy(sphere)
    # check input sphere
    assert sphere.vertex['positions'].shape == (762, 3)
    assert sphere.triangle['indices'].shape == (1520, 3)

    ans = box.boolean_union(sphere)
    assert ans.vertex['positions'].shape == (730, 3)
    assert ans.triangle['indices'].shape == (1384, 3)

    ans = box.boolean_intersection(sphere)
    assert ans.vertex['positions'].shape == (154, 3)
    assert ans.triangle['indices'].shape == (232, 3)

    ans = box.boolean_difference(sphere)
    assert ans.vertex['positions'].shape == (160, 3)
    assert ans.triangle['indices'].shape == (244, 3)
