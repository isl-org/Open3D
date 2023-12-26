# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2023 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    device = o3d.core.Device('CPU:0')
    tum_data = o3d.data.SampleTUMRGBDImage()
    depth = o3d.t.io.read_image(tum_data.depth_path).to(device)
    color = o3d.t.io.read_image(tum_data.color_path).to(device)

    intrinsic = o3d.core.Tensor([[535.4, 0, 320.1], [0, 539.2, 247.6],
                                 [0, 0, 1]])
    rgbd = o3d.t.geometry.RGBDImage(color, depth)

    pcd = o3d.t.geometry.PointCloud.create_from_rgbd_image(rgbd,
                                                           intrinsic,
                                                           depth_scale=5000.0,
                                                           depth_max=10.0)
    o3d.visualization.draw([pcd])
    rgbd_reproj = pcd.project_to_rgbd_image(640,
                                            480,
                                            intrinsic,
                                            depth_scale=5000.0,
                                            depth_max=10.0)

    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(np.asarray(rgbd_reproj.color.to_legacy()))
    axs[1].imshow(np.asarray(rgbd_reproj.depth.to_legacy()))
    plt.show()
