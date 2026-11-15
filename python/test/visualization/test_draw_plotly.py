# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2026 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import numpy as np
import open3d as o3d
import pytest


def _camera(front, **kwargs):
    mesh = o3d.geometry.TriangleMesh.create_box()
    return o3d.visualization.get_plotly_fig([mesh], front=front,
                                            **kwargs).layout.scene.camera


@pytest.mark.parametrize(
    "front",
    [[1, -1, 0], [1, 2, 3], [-1, -2, -3], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
def test_plotly_front_direction(front):
    eye = _camera(front).eye
    actual = np.array([eye.x, eye.y, eye.z])
    expected = 5 * np.asarray(front) / np.linalg.norm(front)
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize("scale", [1., 2., 1e308, 1e-308, np.nextafter(0., 1.)])
def test_plotly_front_scale_invariance(scale):
    eye = _camera(np.array([1., -1., 1.]) * scale).eye
    np.testing.assert_allclose([eye.x, eye.y, eye.z],
                               5 * np.array([1., -1., 1.]) / np.sqrt(3))


@pytest.mark.parametrize(
    "front", [[0, 0, 0], [np.nan, 0, 1], [np.inf, 0, 1], [-np.inf, 0, 1]])
def test_plotly_front_invalid(front):
    with pytest.raises(ValueError, match="finite, nonzero vector"):
        _camera(front)


@pytest.mark.parametrize("readonly", [False, True])
def test_plotly_front_preserves_input(readonly):
    front = np.array([2., -2., 2.])
    original = front.copy()
    front.flags.writeable = not readonly
    _camera(front)
    np.testing.assert_array_equal(front, original)


def test_plotly_front_default():
    assert _camera(None).eye.to_plotly_json() == {}


def test_plotly_front_preserves_zoom_and_center():
    camera = _camera([1, 0, 0], zoom=2, lookat=[0.25, 0.75, 0.5])
    np.testing.assert_allclose(
        [camera.center.x, camera.center.y, camera.center.z], [-0.25, 0.25, 0.])
    np.testing.assert_allclose([camera.eye.x, camera.eye.y, camera.eye.z],
                               [9.75, 0.25, 0.])
