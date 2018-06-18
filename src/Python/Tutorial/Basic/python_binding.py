# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

import numpy as np

def example_help_function():
    import open3d
    help(open3d)
    help(open3d.PointCloud)
    help(open3d.read_point_cloud)

def example_import_function():
    from open3d import read_point_cloud
    pcd = read_point_cloud("../../TestData/ICP/cloud_bin_0.pcd")
    print(pcd)

if __name__ == "__main__":
    example_help_function()
    example_import_function()
