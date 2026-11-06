# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2026 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import numpy as np
import open3d as o3d
import pytest


@pytest.mark.parametrize('sample_factor, expected_count', [(1, 4), (0.5, 2)])
def test_default_point_color_preserves_cloud(sample_factor, expected_count):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.arange(12).reshape(4, 3))

    fig = o3d.visualization.get_plotly_fig([cloud],
                                           point_sample_factor=sample_factor)

    assert not cloud.has_colors()
    assert len(fig.data[0].x) == expected_count
    assert fig.data[0].marker.color == 'red'
