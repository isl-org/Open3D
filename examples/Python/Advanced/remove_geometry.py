# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

# examples/Python/Advanced/remove_geometry.py

import open3d as o3d
import numpy as np
import time
import copy


def visualize_non_blocking(vis, pcds):
    for pcd in pcds:
        vis.update_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()


pcd_orig = o3d.io.read_point_cloud("../../TestData/fragment.pcd")
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
