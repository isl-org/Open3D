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
import time
import os
import sys

pyexample_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
test_data_path = os.path.join(os.path.dirname(pyexample_path), 'test_data')
sys.path.append(pyexample_path)

from open3d_example import read_trajectory


def run_benchmark():
    dataset_path = os.path.join(test_data_path, "RGBD")
    camera_poses = read_trajectory(os.path.join(dataset_path, "odometry.log"))
    camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(
        o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
    volume = o3d.pipelines.integration.UniformTSDFVolume(
        length=4.0,
        resolution=512,
        sdf_trunc=0.04,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    s = time.time()
    for i in range(len(camera_poses)):
        color = o3d.io.read_image(
            os.path.join(dataset_path, "color", "{0:05d}.jpg".format(i)))
        depth = o3d.io.read_image(
            os.path.join(dataset_path, "depth", "{0:05d}.png".format(i)))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, depth, depth_trunc=4.0, convert_rgb_to_intensity=False)
        volume.integrate(
            rgbd,
            camera_intrinsics,
            np.linalg.inv(camera_poses[i].pose),
        )
    time_integrate = time.time() - s

    s = time.time()
    mesh = volume.extract_triangle_mesh()
    time_extract_mesh = time.time() - s

    s = time.time()
    pcd = volume.extract_point_cloud()
    time_extract_pcd = time.time() - s

    return time_integrate, time_extract_mesh, time_extract_pcd


if __name__ == "__main__":
    times = []
    for i in range(10):
        print("Running benchmark {}".format(i))
        time_integrate, time_extract_mesh, time_extract_pcd = run_benchmark()
        times.append([time_integrate, time_extract_mesh, time_extract_pcd])
    avg_times = np.mean(np.array(times), axis=0)

    print("Integrate 512x512x512:", avg_times[0])
    print("Extract mesh:", avg_times[1])
    print("Extract pcd: ", avg_times[2])
