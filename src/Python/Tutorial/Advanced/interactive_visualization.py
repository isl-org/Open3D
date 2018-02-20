
import sys
sys.path.append("../..")
from py3d import *

def draw_registration_result(source, target, transformation):
	source_temp = copy.deepcopy(source)
	target_temp = copy.deepcopy(target)
	source_temp.paint_uniform_color([1, 0.706, 0])
	target_temp.paint_uniform_color([0, 0.651, 0.929])
	source_temp.transform(transformation)
	draw_geometries([source_temp, target_temp])

def pick_points(pcd):
	print("Please pick three points at least.")
	vis = VisualizerWithEditing()
	vis.create_window()
	vis.add_geometry(pcd)
	vis.run()
	vis.destroy_window()
	picked_id = vis.get_picked_points()
	return pcd.points[picked_id,:]

def demo_manual_icp():
	pcd_source = read_point_cloud("../../TestData/ICP/cloud_bin_0.pcd")
	pcd_target = read_point_cloud("../../TestData/ICP/cloud_bin_2.pcd")
	picked_source = pick_points(pcd_source)
	picked_target = pick_points(pcd_target)
	print(picked_source)
	print(picked_target)
	# # compute trans_init
	# # trans_init =
	# reg_p2p = registration_icp(source, target, threshold, trans_init,
	# 		TransformationEstimationPointToPoint())
	# print(reg_p2p)
	# print("Transformation is:")
	# print(reg_p2p.transformation)
	# print("")
	# draw_registration_result(source, target, reg_p2p.transformation)

if __name__ == "__main__":
	demo_manual_icp()
