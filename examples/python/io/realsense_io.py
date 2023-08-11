# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2023 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------
"""Demonstrate RealSense camera discovery and frame capture"""

import open3d as o3d

if __name__ == "__main__":

    o3d.t.io.RealSenseSensor.list_devices()
    rscam = o3d.t.io.RealSenseSensor()
    rscam.start_capture()
    print(rscam.get_metadata())
    for fid in range(5):
        rgbd_frame = rscam.capture_frame()
        o3d.io.write_image(f"color{fid:05d}.jpg", rgbd_frame.color.to_legacy())
        o3d.io.write_image(f"depth{fid:05d}.png", rgbd_frame.depth.to_legacy())
        print("Frame: {}, time: {}s".format(fid, rscam.get_timestamp() * 1e-6))

    rscam.stop_capture()
