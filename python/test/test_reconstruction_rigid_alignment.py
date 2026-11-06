# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2026 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

from pathlib import Path
import runpy

import numpy as np
import pytest


def test_reflection_returns_optimal_proper_rotation():
    pytest.importorskip("cv2")
    pytest.importorskip("matplotlib")
    path = Path(__file__).resolve().parents[
        2] / "examples/python/reconstruction_system/opencv_pose_estimation.py"
    pose = runpy.run_path(str(path))
    source = np.array([[3., -3., 0., 0., 0., 0.], [0., 0., 2., -2., 0., 0.],
                       [0., 0., 0., 0., 1., -1.]])
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    target = rotation @ np.diag([1., 1., -1.]) @ source + [[1.], [2.], [3.]]
    fitted, translation = pose['estimate_3D_transform'](source, target)
    np.testing.assert_allclose(fitted, rotation, atol=1e-12)
    np.testing.assert_allclose(translation[:, 0], [1., 2., 3.])
