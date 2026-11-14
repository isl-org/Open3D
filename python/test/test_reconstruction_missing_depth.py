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


def test_pose_estimation_rejects_missing_depth():
    pytest.importorskip("cv2")
    pytest.importorskip("matplotlib")
    path = Path(__file__).resolve().parents[
        2] / "examples/python/reconstruction_system/opencv_pose_estimation.py"
    pose = runpy.run_path(str(path))
    import open3d as o3d
    rng = np.random.default_rng(24)
    gray = rng.random((144, 192)).astype(np.float32)
    frame = o3d.geometry.RGBDImage()
    frame.color = o3d.geometry.Image(gray)
    frame.depth = o3d.geometry.Image(np.zeros_like(gray))
    intrinsic = o3d.camera.PinholeCameraIntrinsic(192, 144, 130., 130., 96.,
                                                  72.)
    success, transformation = pose['pose_estimation'](frame, frame, intrinsic,
                                                      False)
    assert not success
    np.testing.assert_array_equal(transformation, np.eye(4))
