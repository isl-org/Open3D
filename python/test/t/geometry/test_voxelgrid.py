# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# The MIT License (MIT)
#
# Copyright (c) 2020 www.open3d.org
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

import sys
import os

test_path = os.path.dirname(os.path.realpath(__file__)) + "/../../"
sys.path.append(test_path)
test_data_path = test_path + "../../examples/test_data/"

from open3d_test import list_devices


class CameraPose:

    def __init__(self, meta, mat):
        self.metadata = meta
        self.pose = mat

    def __str__(self):
        return 'Metadata : ' + ' '.join(map(str, self.metadata)) + '\n' + \
            "Pose : " + "\n" + np.array_str(self.pose)


def read_trajectory(filename):
    traj = []
    with open(filename, 'r') as f:
        metastr = f.readline()
        while metastr:
            metadata = list(map(int, metastr.split()))
            mat = np.zeros(shape=(4, 4))
            for i in range(4):
                matstr = f.readline()
                mat[i, :] = np.fromstring(matstr, dtype=float, sep=' \t')
            traj.append(CameraPose(metadata, mat))
            metastr = f.readline()
    return traj


@pytest.mark.parametrize("device", list_devices())
def test_integration(device):
    voxel_size = 0.008  # voxel resolution in meter
    sdf_trunc = 0.04  # truncation distance in meter
    block_resolution = 16  # 16^3 voxel blocks
    initial_block_count = 1000  # initially allocated number of voxel blocks

    volume = o3d.t.geometry.TSDFVoxelGrid(
        {
            'tsdf': o3d.core.Dtype.Float32,
            'weight': o3d.core.Dtype.UInt16,
            'color': o3d.core.Dtype.UInt16
        },
        voxel_size=voxel_size,
        sdf_trunc=sdf_trunc,
        block_resolution=block_resolution,
        block_count=initial_block_count,
        device=device)

    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)

    intrinsic = o3d.core.Tensor(intrinsic.intrinsic_matrix,
                                o3d.core.Dtype.Float32, device)

    camera_poses = read_trajectory(test_data_path + "RGBD/odometry.log")

    for i in range(len(camera_poses)):
        color = o3d.io.read_image(test_data_path +
                                  "RGBD/color/{:05d}.jpg".format(i))
        color = o3d.t.geometry.Image.from_legacy_image(color, device=device)

        depth = o3d.io.read_image(test_data_path +
                                  "RGBD/depth/{:05d}.png".format(i))
        depth = o3d.t.geometry.Image.from_legacy_image(depth, device=device)

        extrinsic = o3d.core.Tensor(np.linalg.inv(camera_poses[i].pose),
                                    o3d.core.Dtype.Float32, device)
        volume.integrate(depth, color, intrinsic, extrinsic, 1000.0, 3.0)
        if i == len(camera_poses) - 1:
            vertexmap, _, _ = volume.raycast(intrinsic, extrinsic,
                                             depth.columns, depth.rows, 50, 0.1,
                                             3.0, min(i * 1.0, 3.0))
            vertexmap_gt = np.load(
                test_data_path +
                "open3d_downloads/RGBD/raycast_vtx_{:03d}.npy".format(i))
            discrepancy_count = ((vertexmap.cpu().numpy() - vertexmap_gt) >
                                 1e-5).sum()
            # Be tolerant to numerical differences
            assert discrepancy_count / vertexmap_gt.size < 1e-3

    pcd = volume.extract_surface_points().to_legacy_pointcloud()
    pcd_gt = o3d.io.read_point_cloud(test_data_path +
                                     "RGBD/example_tsdf_pcd.ply")

    n_pcd = len(pcd.points)
    n_pcd_gt = len(pcd_gt.points)
    assert np.abs(n_pcd - n_pcd_gt) < 3

    result = o3d.pipelines.registration.evaluate_registration(
        pcd, pcd_gt, voxel_size, np.identity(4))
    assert result.fitness > 1 - 1e-5
    assert result.inlier_rmse < 1e-5
