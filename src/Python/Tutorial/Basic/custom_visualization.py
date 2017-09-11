# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

import sys
sys.path.append("../..")
from py3d import *

def custom_draw_geometry(pcd):
	# The following code achieves the same effect as:
	# DrawGeometries([pcd])
	vis = Visualizer()
	vis.CreateWindow()
	vis.AddGeometry(pcd)
	vis.Run()
	vis.DestroyWindow()	

def rotate_view(vis):
	ctr = vis.GetViewControl()
	ctr.Rotate(10.0, 0.0)
	return False

def custom_draw_geometry_with_rotation(pcd):
	vis = Visualizer()
	vis.CreateWindow()
	vis.AddGeometry(pcd)
	vis.RegisterAnimationCallback(rotate_view)
	vis.Run()
	vis.DestroyWindow()	

if __name__ == "__main__":
	pcd = ReadPointCloud("../../TestData/fragment.ply")

	print("1. Customized visualization to mimic DrawGeometry")
	custom_draw_geometry(pcd)

	print("2. Customized visualization with a rotating view")
	custom_draw_geometry_with_rotation(pcd)
