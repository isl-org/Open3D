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
import copy


def visualize_non_blocking(vis, pcds):
    for pcd in pcds:
        vis.update_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()


pcd_data = o3d.data.PCDPointCloud()
pcd_orig = o3d.io.read_point_cloud(pcd_data.path)
flip_transform = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
pcd_orig.transform(flip_transform)
n_pcd = 5
pcds = []
for i in range(n_pcd):
    pcds.append(copy.deepcopy(pcd_orig))
    trans = np.identity(4)
    trans[:3, 3] = [3 * i, 0, 0]
    pcds[i].transform(trans)

vis = o3d.visualization.Visualizer()
vis.create_window()
start_time = time.time()
added = [False] * n_pcd

curr_sec = int(time.time() - start_time)
prev_sec = curr_sec - 1

while True:
    curr_sec = int(time.time() - start_time)
    if curr_sec - prev_sec == 1:
        prev_sec = curr_sec

        for i in range(n_pcd):
            if curr_sec % (n_pcd * 2) == i and not added[i]:
                vis.add_geometry(pcds[i])
                added[i] = True
                print("Adding %d" % i)
            if curr_sec % (n_pcd * 2) == (i + n_pcd) and added[i]:
                vis.remove_geometry(pcds[i])
                added[i] = False
                print("Removing %d" % i)

    visualize_non_blocking(vis, pcds)
