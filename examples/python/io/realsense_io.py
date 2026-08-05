# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------
"""Demonstrate RealSense camera discovery and frame capture. An RS bag file is
used if a RealSense camera is not available. Captured frames are
displayed as a live point cloud. Also frames are saved to ./capture/{color,depth}
folders.

Usage:

    - Display live point cloud from RS camera:
        python realsense_io.py

    - Display live point cloud from RS bag file:
        python realsense_io.py rgbd.bag

    If no device is detected and no bag file is supplied, uses the Open3D
    example JackJackL515Bag dataset.
"""

import sys
import os
from concurrent.futures import ThreadPoolExecutor
import open3d as o3d
import open3d.t.io as io3d
from open3d.t.geometry import PointCloud
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

DEPTH_MAX = 3


def point_cloud_video(executor, rgbd_frame, mdata, timestamp, o3dvis):
    """Update window to display the next point cloud frame."""
    app = gui.Application.instance
    update_flag = rendering.Scene.UPDATE_POINTS_FLAG | rendering.Scene.UPDATE_COLORS_FLAG

    executor.submit(io3d.write_image,
                    f"capture/color/{point_cloud_video.fid:05d}.jpg",
                    rgbd_frame.color)
    executor.submit(io3d.write_image,
                    f"capture/depth/{point_cloud_video.fid:05d}.png",
                    rgbd_frame.depth)
    print(f"Frame: {point_cloud_video.fid}, timestamp: {timestamp * 1e-6:.3f}s",
          end="\r")
    if point_cloud_video.fid == 0:
        # Start with a dummy max sized point cloud to allocate GPU buffers
        # for update_geometry()
        max_pts = rgbd_frame.color.rows * rgbd_frame.color.columns
        pcd = PointCloud(o3d.core.Tensor.zeros((max_pts, 3)))
        pcd.paint_uniform_color([1., 1., 1.])
        app.post_to_main_thread(o3dvis,
                                lambda: o3dvis.add_geometry("Point Cloud", pcd))
    pcd = PointCloud.create_from_rgbd_image(
        rgbd_frame,
        # Intrinsic matrix: Tensor([[fx, 0., cx], [0., fy, cy], [0., 0., 1.]])
        mdata.intrinsics.intrinsic_matrix,
        depth_scale=mdata.depth_scale,
        depth_max=DEPTH_MAX)
    # GUI operations MUST run in the main thread.
    app.post_to_main_thread(
        o3dvis, lambda: o3dvis.update_geometry("Point Cloud", pcd, update_flag))
    point_cloud_video.fid += 1


point_cloud_video.fid = 0


def pcd_video_from_camera(executor, o3dvis):
    """Show point cloud video coming from a RealSense camera. Save frames to
    disk in capture/{color,depth} folders.
    """
    rscam = io3d.RealSenseSensor()
    rscam.start_capture()
    mdata = rscam.get_metadata()
    print(mdata)
    os.makedirs("capture/color")
    os.makedirs("capture/depth")
    rgbd_frame_future = executor.submit(rscam.capture_frame)

    def on_window_close():
        nonlocal rscam, executor
        executor.shutdown()
        rscam.stop_capture()
        return True  # OK to close window

    o3dvis.set_on_close(on_window_close)

    while True:
        rgbd_frame = rgbd_frame_future.result()
        # Run each IO operation in the threadpool
        rgbd_frame_future = executor.submit(rscam.capture_frame)
        point_cloud_video(executor, rgbd_frame, mdata, rscam.get_timestamp(),
                          o3dvis)


def pcd_video_from_bag(rsbagfile, executor, o3dvis):
    """Show point cloud video coming from a RealSense bag file. Save frames to
    disk in capture/{color,depth} folders.
    """
    rsbag = io3d.RSBagReader.create(rsbagfile)
    if not rsbag.is_opened():
        raise RuntimeError(f"RS bag file {rsbagfile} could not be opened.")
    mdata = rsbag.metadata
    print(mdata)
    os.makedirs("capture/color")
    os.makedirs("capture/depth")

    def on_window_close():
        nonlocal rsbag, executor
        executor.shutdown()
        rsbag.close()
        return True  # OK to close window

    o3dvis.set_on_close(on_window_close)

    rgbd_frame = rsbag.next_frame()
    while not rsbag.is_eof():
        # Run each IO operation in the threadpool
        rgbd_frame_future = executor.submit(rsbag.next_frame)
        point_cloud_video(executor, rgbd_frame, mdata, rsbag.get_timestamp(),
                          o3dvis)
        rgbd_frame = rgbd_frame_future.result()


def main():
    if os.path.exists("capture"):
        raise RuntimeError(
            "Frames saving destination folder 'capture' already exists. Please move it."
        )

    # Initialize app and create GUI window
    app = gui.Application.instance
    app.initialize()
    o3dvis = o3d.visualization.O3DVisualizer("Open3D: PointCloud video", 1024,
                                             768)
    o3dvis.show_axes = True
    # set view: fov, eye, lookat, up
    o3dvis.setup_camera(45, [0., 0., 0.], [0., 0., -1.], [0., -1., 0.])
    app.add_window(o3dvis)

    have_cam = io3d.RealSenseSensor.list_devices()
    have_bag = (len(sys.argv) == 2)

    with ThreadPoolExecutor(max_workers=4) as executor:
        # Run IO and compute in threadpool
        if have_bag:
            executor.submit(pcd_video_from_bag, sys.argv[1], executor, o3dvis)
        elif have_cam:
            executor.submit(pcd_video_from_camera, executor, o3dvis)
        else:
            executor.submit(pcd_video_from_bag,
                            o3d.data.JackJackL515Bag().path, executor, o3dvis)

        app.run()  # GUI runs in the main thread.


if __name__ == "__main__":
    main()
