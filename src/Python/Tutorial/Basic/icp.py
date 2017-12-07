# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

import sys
sys.path.append("../..")
from py3d import *
from trajectory_io import *

if __name__ == "__main__":

	traj = read_trajectory("../../TestData/ICP/init.log")
	pcds = []
	threshold = 0.02
	for i in range(3):
		pcds.append(read_point_cloud("../../TestData/ICP/cloud_bin_{:d}.pcd".format(i)))

	for reg in traj:
		target = pcds[reg.metadata[0]]
		source = pcds[reg.metadata[1]]
		trans = reg.pose
		evaluation_init = evaluate_registration(source, target, threshold, trans)
		print(evaluation_init)

		print("Apply point-to-point ICP")
		reg_p2p = registration_icp(source, target, threshold, trans, TransformationEstimationPointToPoint())
		print(reg_p2p)
		print("Transformation is:")
		print(reg_p2p.transformation)

		print("Apply point-to-plane ICP")
		reg_p2l = registration_icp(source, target, threshold, trans, TransformationEstimationPointToPlane())
		print(reg_p2l)
		print("Transformation is:")
		print(reg_p2l.transformation)
		print("")

	print("")
