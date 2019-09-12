# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

# examples/Python/Benchmark/benchmark_tsdf.py

import open3d as o3d
import numpy as np
import time
import os
import sys

pwd = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(pwd, '..', 'Advanced'))
from trajectory_io import read_trajectory


def run_benchmark():
    dataset_path = os.path.join(pwd, "..", "..", "TestData", "RGBD")
    camera_poses = read_trajectory(os.path.join(dataset_path, "odometry.log"))
    camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(
        o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
    volume = o3d.integration.UniformTSDFVolume(
        length=4.0,
        resolution=512,
        sdf_trunc=0.04,
        color_type=o3d.integration.TSDFVolumeColorType.RGB8,
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
        print(f"Running benchmark {i}")
        time_integrate, time_extract_mesh, time_extract_pcd = run_benchmark()
        times.append([time_integrate, time_extract_mesh, time_extract_pcd])
    avg_times = np.mean(np.array(times), axis=0)

    print("Integrate 512x512x512:", avg_times[0])
    print("Extract mesh:", avg_times[1])
    print("Extract pcd: ", avg_times[2])
