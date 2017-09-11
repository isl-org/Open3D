# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

import sys, os
sys.path.append("../..")
from py3d import *
import numpy as np

def custom_draw_geometry(pcd):
	# The following code achieves the same effect as:
	# DrawGeometries([pcd])
	vis = Visualizer()
	vis.CreateWindow()
	vis.AddGeometry(pcd)
	vis.Run()
	vis.DestroyWindow()	

def custom_draw_geometry_with_rotation(pcd):
	def rotate_view(vis):
		ctr = vis.GetViewControl()
		ctr.Rotate(10.0, 0.0)
		return False
	vis = Visualizer()
	vis.CreateWindow()
	vis.AddGeometry(pcd)
	vis.RegisterAnimationCallback(rotate_view)
	vis.Run()
	vis.DestroyWindow()	

def custom_draw_geometry_with_black_background(pcd):
	vis = Visualizer()
	vis.CreateWindow()
	vis.AddGeometry(pcd)
	vis.GetRenderOption().LoadFromJSON("../../TestData/renderoption.json")
	vis.Run()
	vis.DestroyWindow()	

def custom_draw_geometry_with_key_callback(pcd):
	def change_background_to_black(vis):
		opt = vis.GetRenderOption()
		opt.background_color = np.asarray([0, 0, 0])
		return False
	vis = VisualizerWithKeyCallback()
	vis.CreateWindow()
	vis.AddGeometry(pcd)
	vis.RegisterKeyCallback(ord("K"), change_background_to_black)
	vis.Run()
	vis.DestroyWindow()

def custom_draw_geometry_with_camera_trajectory(pcd):
	custom_draw_geometry_with_camera_trajectory.index = -1
	custom_draw_geometry_with_camera_trajectory.trajectory =\
			ReadPinholeCameraTrajectory("../../TestData/camera_trajectory.json")
	custom_draw_geometry_with_camera_trajectory.vis = Visualizer()
	if not os.path.exists("image/"):
		os.makedirs("image/")
	def move_forward(vis):
		# This function is called within the Visualizer::Run() loop
		# The Run loop calls the function, then re-render
		# So the sequence in this function is to:
		# 1. Capture frame
		# 2. index++, check ending criteria
		# 3. Set camera
		# 4. (Re-render)
		ctr = vis.GetViewControl()
		glb = custom_draw_geometry_with_camera_trajectory
		if glb.index >= 0:
			print("Capture image #{:05d}".format(glb.index))
			vis.CaptureScreenImage("image/{:05d}.png".format(glb.index), False)
		glb.index = glb.index + 1
		if glb.index < len(glb.trajectory.extrinsic):
			ctr.ConvertFromPinholeCameraParameters(glb.trajectory.intrinsic,\
					glb.trajectory.extrinsic[glb.index])
		else:
			custom_draw_geometry_with_camera_trajectory.vis.\
					RegisterAnimationCallback(None)
		return False
	vis = custom_draw_geometry_with_camera_trajectory.vis
	vis.CreateWindow()
	vis.AddGeometry(pcd)
	vis.RegisterAnimationCallback(move_forward)
	vis.Run(True)
	vis.DestroyWindow()	

if __name__ == "__main__":
	pcd = ReadPointCloud("../../TestData/fragment.ply")

	print("1. Customized visualization to mimic DrawGeometry")
	custom_draw_geometry(pcd)

	print("2. Customized visualization with a rotating view")
	custom_draw_geometry_with_rotation(pcd)

	print("3. Customized visualization loading a black background")
	custom_draw_geometry_with_black_background(pcd)

	print("4. Customized visualization loading a black background when press K")
	custom_draw_geometry_with_key_callback(pcd)

	print("5. Customized visualization playing a camera trajectory")
	custom_draw_geometry_with_camera_trajectory(pcd)
