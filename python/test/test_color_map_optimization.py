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
import re
import os
from open3d_test import download_fountain_dataset


def load_fountain_dataset():
    rgbd_images = []
    fountain_rgbd_dataset = o3d.data.SampleFountainRGBDDataset()
    for i in range(len(fountain_rgbd_dataset.depth_paths)):
        depth = o3d.io.read_image(fountain_rgbd_dataset.depth_paths[i])
        color = o3d.io.read_image(fountain_rgbd_dataset.color_paths[i])
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, depth, convert_rgb_to_intensity=False)
        rgbd_images.append(rgbd_image)

    camera_trajectory = o3d.io.read_pinhole_camera_trajectory(
        fountain_rgbd_dataset.keyframe_poses_log_path)
    mesh = o3d.io.read_triangle_mesh(fountain_rgbd_dataset.reconstruction_path)

    return mesh, rgbd_images, camera_trajectory


def test_color_map():
    """
    Hard-coded values are from the 0.12 release. We expect the values to match
    exactly when OMP_NUM_THREADS=1. If more threads are used, there could be
    some small numerical differences.
    """
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)

    # Load dataset
    mesh, rgbd_images, camera_trajectory = load_fountain_dataset()

    # Computes averaged color without optimization, for debugging
    mesh, camera_trajectory = o3d.pipelines.color_map.run_rigid_optimizer(
        mesh, rgbd_images, camera_trajectory,
        o3d.pipelines.color_map.RigidOptimizerOption(maximum_iteration=0))
    vertex_mean = np.mean(np.asarray(mesh.vertex_colors), axis=0)
    extrinsic_mean = np.array(
        [c.extrinsic for c in camera_trajectory.parameters]).mean(axis=0)
    np.testing.assert_allclose(vertex_mean,
                               np.array([0.40322907, 0.37276872, 0.54375919]),
                               rtol=1e-5)
    np.testing.assert_allclose(
        extrinsic_mean,
        np.array([[0.77003829, -0.10813595, 0.06467495, -0.56212008],
                  [0.19100387, 0.86225833, -0.14664845, -0.81434887],
                  [-0.05557141, 0.16504166, 0.82036438, 0.27867426],
                  [0., 0., 0., 1.]]),
        rtol=1e-5)

    # Rigid Optimization
    mesh, camera_trajectory = o3d.pipelines.color_map.run_rigid_optimizer(
        mesh, rgbd_images, camera_trajectory,
        o3d.pipelines.color_map.RigidOptimizerOption(maximum_iteration=10))

    vertex_mean = np.mean(np.asarray(mesh.vertex_colors), axis=0)
    extrinsic_mean = np.array(
        [c.extrinsic for c in camera_trajectory.parameters]).mean(axis=0)
    np.testing.assert_allclose(vertex_mean,
                               np.array([0.40294861, 0.37250299, 0.54338467]),
                               rtol=1e-5)
    np.testing.assert_allclose(
        extrinsic_mean,
        np.array([[0.7699379, -0.10768808, 0.06543989, -0.56320637],
                  [0.19119488, 0.8619734, -0.14717332, -0.8137762],
                  [-0.05608781, 0.16546427, 0.81995183, 0.27725451],
                  [0., 0., 0., 1.]]),
        rtol=1e-5)

    # Non-rigid Optimization
    mesh, camera_trajectory = o3d.pipelines.color_map.run_non_rigid_optimizer(
        mesh, rgbd_images, camera_trajectory,
        o3d.pipelines.color_map.NonRigidOptimizerOption(maximum_iteration=10))

    vertex_mean = np.mean(np.asarray(mesh.vertex_colors), axis=0)
    extrinsic_mean = np.array(
        [c.extrinsic for c in camera_trajectory.parameters]).mean(axis=0)
    np.testing.assert_allclose(vertex_mean,
                               np.array([0.4028204, 0.37237733, 0.54322786]),
                               rtol=1e-5)
    np.testing.assert_allclose(
        extrinsic_mean,
        np.array([[0.76967962, -0.10824218, 0.0674025, -0.56381652],
                  [0.19129921, 0.86245618, -0.14634957, -0.81500831],
                  [-0.05765316, 0.16483281, 0.82054672, 0.27526268],
                  [0., 0., 0., 1.]]),
        rtol=1e-5)
