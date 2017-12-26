# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

import sys
sys.path.append("../..")
from py3d import *

def optimize_posegraph(pose_graph_name, pose_graph_optmized_name):
	# to display messages from global_optimization
	set_verbosity_level(VerbosityLevel.Debug)
	method = GlobalOptimizationLevenbergMarquardt()
	criteria = GlobalOptimizationConvergenceCriteria()
	option = GlobalOptimizationOption(
			max_correspondence_distance = 0.07,
			edge_prune_threshold = 0.25,
			reference_node = 0)
	pose_graph = read_pose_graph(pose_graph_name)
	global_optimization(pose_graph, method, criteria, option)
	write_pose_graph(pose_graph_optmized_name, pose_graph)
	set_verbosity_level(VerbosityLevel.Error)
