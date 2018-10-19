# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

import numpy as np
from open3d import *

if __name__ == "__main__":

    pcd = read_point_cloud("../../TestData/fragment.ply")
    min_cube_size = 0.05
    print("\nOriginal, # of points %d" % (np.asarray(pcd.points).shape[0]))
    pcd_down = voxel_down_sample(pcd, min_cube_size)
    print("\nScale %f, # of points %d" % \
            (min_cube_size, np.asarray(pcd_down.points).shape[0]))
    min_bound = pcd_down.get_min_bound() - min_cube_size * 0.5
    max_bound = pcd_down.get_max_bound() + min_cube_size * 0.5

    pcd_curr = pcd_down
    num_scales = 3
    for i in range(1, num_scales):
        multiplier = pow(2, i)
        pcd_curr_down, cubic_id = voxel_down_sample_and_trace(pcd_curr,
            multiplier * min_cube_size, min_bound, max_bound, False)
        print("\nScale %f, # of points %d" % (multiplier * min_cube_size,
                np.asarray(pcd_curr_down.points).shape[0]))
        print("Downsampled points (the first 10 points)")
        print(np.asarray(pcd_curr_down.points)[:10,:])
        print("Index (the first 10 indices)")
        print(np.asarray(cubic_id)[:10,:])
        pcd_curr = pcd_curr_down
