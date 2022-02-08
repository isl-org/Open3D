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

import numpy as np
import open3d as o3d
import threading
import time

CLOUD_NAME = "points"


def main():
    MultiWinApp().run()


class MultiWinApp:

    def __init__(self):
        self.is_done = False
        self.n_snapshots = 0
        self.cloud = None
        self.main_vis = None
        self.snapshot_pos = None

    def run(self):
        app = o3d.visualization.gui.Application.instance
        app.initialize()

        self.main_vis = o3d.visualization.O3DVisualizer(
            "Open3D - Multi-Window Demo")
        self.main_vis.add_action("Take snapshot in new window",
                                 self.on_snapshot)
        self.main_vis.set_on_close(self.on_main_window_closing)

        app.add_window(self.main_vis)
        self.snapshot_pos = (self.main_vis.os_frame.x, self.main_vis.os_frame.y)

        threading.Thread(target=self.update_thread).start()

        app.run()

    def on_snapshot(self, vis):
        self.n_snapshots += 1
        self.snapshot_pos = (self.snapshot_pos[0] + 50,
                             self.snapshot_pos[1] + 50)
        title = "Open3D - Multi-Window Demo (Snapshot #" + str(
            self.n_snapshots) + ")"
        new_vis = o3d.visualization.O3DVisualizer(title)
        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = "defaultUnlit"
        new_vis.add_geometry(CLOUD_NAME + " #" + str(self.n_snapshots),
                             self.cloud, mat)
        new_vis.reset_camera_to_default()
        bounds = self.cloud.get_axis_aligned_bounding_box()
        extent = bounds.get_extent()
        new_vis.setup_camera(60, bounds.get_center(),
                             bounds.get_center() + [0, 0, -3], [0, -1, 0])
        o3d.visualization.gui.Application.instance.add_window(new_vis)
        new_vis.os_frame = o3d.visualization.gui.Rect(self.snapshot_pos[0],
                                                      self.snapshot_pos[1],
                                                      new_vis.os_frame.width,
                                                      new_vis.os_frame.height)

    def on_main_window_closing(self):
        self.is_done = True
        return True  # False would cancel the close

    def update_thread(self):
        # This is NOT the UI thread, need to call post_to_main_thread() to update
        # the scene or any part of the UI.
        pcd_data = o3d.data.DemoICPPointClouds()
        self.cloud = o3d.io.read_point_cloud(pcd_data.paths[0])
        bounds = self.cloud.get_axis_aligned_bounding_box()
        extent = bounds.get_extent()

        def add_first_cloud():
            mat = o3d.visualization.rendering.MaterialRecord()
            mat.shader = "defaultUnlit"
            self.main_vis.add_geometry(CLOUD_NAME, self.cloud, mat)
            self.main_vis.reset_camera_to_default()
            self.main_vis.setup_camera(60, bounds.get_center(),
                                       bounds.get_center() + [0, 0, -3],
                                       [0, -1, 0])

        o3d.visualization.gui.Application.instance.post_to_main_thread(
            self.main_vis, add_first_cloud)

        while not self.is_done:
            time.sleep(0.1)

            # Perturb the cloud with a random walk to simulate an actual read
            pts = np.asarray(self.cloud.points)
            magnitude = 0.005 * extent
            displacement = magnitude * (np.random.random_sample(pts.shape) -
                                        0.5)
            new_pts = pts + displacement
            self.cloud.points = o3d.utility.Vector3dVector(new_pts)

            def update_cloud():
                # Note: if the number of points is less than or equal to the
                #       number of points in the original object that was added,
                #       using self.scene.update_geometry() will be faster.
                #       Requires that the point cloud be a t.PointCloud.
                self.main_vis.remove_geometry(CLOUD_NAME)
                mat = o3d.visualization.rendering.MaterialRecord()
                mat.shader = "defaultUnlit"
                self.main_vis.add_geometry(CLOUD_NAME, self.cloud, mat)

            if self.is_done:  # might have changed while sleeping
                break
            o3d.visualization.gui.Application.instance.post_to_main_thread(
                self.main_vis, update_cloud)


if __name__ == "__main__":
    main()
