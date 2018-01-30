# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

import sys
sys.path.append("../..")
from py3d import *
import numpy as np

if __name__ == "__main__":

	set_verbosity_level(VerbosityLevel.Debug)

	print("")
	print("Parameters for PoseGraph optimization ...")
	method = GlobalOptimizationLevenbergMarquardt()
	criteria = GlobalOptimizationConvergenceCriteria()
	option = GlobalOptimizationOption()
	print("")
	print(method)
	print(criteria)
	print(option)
	print("")

	print("Optimizing Fragment PoseGraph using py3d ...")
	data_path = "../../TestData/GraphOptimization/"
	pose_graph_fragment = read_pose_graph(data_path +
			"pose_graph_example_fragment.json")
	print(pose_graph_fragment)
	global_optimization(pose_graph_fragment, method, criteria, option)
	write_pose_graph(data_path +
			"pose_graph_example_fragment_optimized.json", pose_graph_fragment)
	print("")

	print("Optimizing Global PoseGraph using py3d ...")
	pose_graph_global = read_pose_graph(data_path +
			"pose_graph_example_global.json")
	print(pose_graph_global)
	global_optimization(pose_graph_global, method, criteria, option)
	write_pose_graph(data_path +
			"pose_graph_example_global_optimized.json", pose_graph_global)
	print("")
