# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

import sys
sys.path.append("../..")
sys.path.append("../Utility")
from py3d import *
from common import *

def run_posegraph_optimization(pose_graph_name, pose_graph_optmized_name,
		max_correspondence_distance):
	# to display messages from global_optimization
	set_verbosity_level(VerbosityLevel.Debug)
	method = GlobalOptimizationLevenbergMarquardt()
	criteria = GlobalOptimizationConvergenceCriteria()
	option = GlobalOptimizationOption(
			max_correspondence_distance = max_correspondence_distance,
			edge_prune_threshold = 0.25,
			reference_node = 0)
	pose_graph = read_pose_graph(pose_graph_name)
	global_optimization(pose_graph, method, criteria, option)
	write_pose_graph(pose_graph_optmized_name, pose_graph)
	set_verbosity_level(VerbosityLevel.Error)


def optimize_posegraph_for_fragment(path_dataset, fragment_id):
	pose_graph_name = path_dataset + template_fragment_posegraph % fragment_id
	pose_graph_optmized_name = path_dataset + \
			template_fragment_posegraph_optimized % fragment_id
	run_posegraph_optimization(pose_graph_name, pose_graph_optmized_name,
			max_correspondence_distance = 0.03)


def optimize_posegraph_for_scene(path_dataset):
	pose_graph_name = path_dataset + template_global_posegraph
	pose_graph_optmized_name = path_dataset + \
			template_global_posegraph_optimized
	run_posegraph_optimization(pose_graph_name, pose_graph_optmized_name,
			max_correspondence_distance = 0.03)
