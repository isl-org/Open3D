# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

import sys
sys.path.append("../..")
from py3d import *
import numpy as np

if __name__ == "__main__":

	SetVerbosityLevel(VerbosityLevel.Debug)

	print("")
	print("Parameters for PoseGraph optimization ...")
	method = GlobalOptimizationLevenbergMarquardt()
	criteria = GlobalOptimizationConvergenceCriteria()
	line_process_option = GlobalOptimizationLineProcessOption()
	print("")
	print(method)
	print(criteria)
	print(line_process_option)
	print("")

	print("Optimizing Fragment PoseGraph using py3d ...")
	pose_graph_fragment = ReadPoseGraph(
			"../../TestData/GraphOptimization/pose_graph_example_fragment.json")
	print(pose_graph_fragment)
	GlobalOptimization(pose_graph_fragment, method, criteria, line_process_option)
	WritePoseGraph(
			"../../TestData/GraphOptimization/pose_graph_example_fragment_optimized.json",
			pose_graph_fragment)
	print("")

	print("Optimizing Global PoseGraph using py3d ...")
	pose_graph_global = ReadPoseGraph(
			"../../TestData/GraphOptimization/pose_graph_example_global.json")
	print(pose_graph_global)
	GlobalOptimization(pose_graph_global, method, criteria, line_process_option)
	WritePoseGraph(
			"../../TestData/GraphOptimization/pose_graph_example_global_optimized.json",
			pose_graph_global)
	print("")
