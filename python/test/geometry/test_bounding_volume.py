# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2026 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import numpy as np
import open3d as o3d
import pytest

_CENTER = np.array([1.0, 2.0, 3.0])
_EXTENT = np.array([2.0, 4.0, 6.0])


def _rotation():
    """Rotation built from the 30, 45 and 60 degree Euler angles."""
    return o3d.geometry.get_rotation_matrix_from_xyz(
        (np.pi / 6.0, np.pi / 4.0, np.pi / 3.0))


def _transform(linear, translation):
    transformation = np.eye(4)
    transformation[:3, :3] = linear
    transformation[:3, 3] = translation
    return transformation


def _box():
    return o3d.geometry.OrientedBoundingBox(_CENTER, _rotation(), _EXTENT)


def test_oriented_bounding_box_transform_matches_points():
    """The box corners follow the transform the way point coordinates do."""
    transformation = _transform(2.0 * _rotation(), [-4.0, 5.0, 0.5])
    corners = np.asarray(_box().get_box_points())
    expected = corners @ transformation[:3, :3].T + transformation[:3, 3]

    box = _box()
    box.transform(transformation)

    np.testing.assert_allclose(np.asarray(box.get_box_points()),
                               expected,
                               atol=1e-9)
    np.testing.assert_allclose(box.extent, 2.0 * _EXTENT, atol=1e-9)


def test_oriented_bounding_box_transform_matches_point_cloud():
    """A box and a point cloud of its corners transform consistently."""
    transformation = _transform(_rotation(), [-4.0, 5.0, 0.5])
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(
        np.asarray(_box().get_box_points()))
    cloud.transform(transformation)

    box = _box()
    box.transform(transformation)

    np.testing.assert_allclose(np.asarray(box.get_box_points()),
                               np.asarray(cloud.points),
                               atol=1e-9)


@pytest.mark.parametrize("linear", [
    np.diag([1.0, 2.0, 3.0]),
    np.array([[1.0, 0.5, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
    -np.eye(3),
])
def test_oriented_bounding_box_transform_rejects_non_similarity(linear):
    box = _box()
    with pytest.raises(RuntimeError):
        box.transform(_transform(linear, [0.0, 0.0, 0.0]))


def test_oriented_bounding_ellipsoid_transform():
    transformation = _transform(2.0 * _rotation(), [-4.0, 5.0, 0.5])
    ellipsoid = o3d.geometry.OrientedBoundingEllipsoid(_CENTER, np.eye(3),
                                                       [1.0, 2.0, 3.0])
    ellipsoid.transform(transformation)

    np.testing.assert_allclose(ellipsoid.center,
                               transformation[:3, :3] @ _CENTER +
                               transformation[:3, 3],
                               atol=1e-9)
    np.testing.assert_allclose(ellipsoid.R, _rotation(), atol=1e-9)
    np.testing.assert_allclose(ellipsoid.radii, [2.0, 4.0, 6.0], atol=1e-9)
