# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

from open3d import *
import numpy as np
import copy

if __name__ == "__main__":
    set_verbosity_level(VerbosityLevel.Debug)
    source_raw = read_point_cloud("../../TestData/ICP/cloud_bin_0.pcd")
    target_raw = read_point_cloud("../../TestData/ICP/cloud_bin_1.pcd")
    source = voxel_down_sample(source_raw, voxel_size = 0.02)
    target = voxel_down_sample(target_raw, voxel_size = 0.02)
    trans = [[0.862, 0.011, -0.507,  0.0],
            [-0.139, 0.967, -0.215,  0.7],
            [0.487, 0.255,  0.835, -1.4],
            [0.0, 0.0, 0.0, 1.0]]
    source.transform(trans)

    flip_transform = [[1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]]
    source.transform(flip_transform)
    target.transform(flip_transform)

    vis = Visualizer()
    vis.create_window()
    vis.add_geometry(source)
    vis.add_geometry(target)
    threshold = 0.05
    icp_iteration = 100
    save_image = False

    for i in range(icp_iteration):
        reg_p2l = registration_icp(source, target, threshold,
                np.identity(4), TransformationEstimationPointToPlane(),
                ICPConvergenceCriteria(max_iteration = 1))
        source.transform(reg_p2l.transformation)
        vis.update_geometry()
        vis.reset_view_point(True)
        vis.poll_events()
        vis.update_renderer()
        if save_image:
            vis.capture_screen_image("temp_%04d.jpg" % i)
    vis.destroy_window()
