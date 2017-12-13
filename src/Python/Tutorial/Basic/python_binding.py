# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

import sys
import numpy as np
sys.path.append("../..")

def example_help_function():
	import py3d as py3d
	help(py3d)
	help(py3d.PointCloud)
	help(py3d.read_point_cloud)

def example_import_function():
	from py3d import read_point_cloud
	pcd = read_point_cloud("../../TestData/ICP/cloud_bin_0.pcd")
	print(pcd)

if __name__ == "__main__":
	example_help_function()
	example_import_function()
