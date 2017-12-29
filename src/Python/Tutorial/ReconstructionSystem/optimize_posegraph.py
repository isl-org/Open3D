# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

import sys
sys.path.append("../..")
from py3d import *


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


def optimize_posegraph_for_fragment(path_dataset, fragment_id, pose_graph):
	pose_graph_name = path_dataset + \
			"/fragments/fragments_%03d.json" % fragment_id
	write_pose_graph(pose_graph_name, pose_graph)
	pose_graph_optmized_name = path_dataset + \
			"/fragments/fragments_opt_%03d.json" % fragment_id
	run_posegraph_optimization(pose_graph_name, pose_graph_optmized_name,
			max_correspondence_distance = 0.075)


def optimize_posegraph_for_scene(path_dataset, pose_graph):
	pose_graph_name = path_dataset + \
			"/fragments/global_registration.json"
	write_pose_graph(pose_graph_name, pose_graph)
	pose_graph_optmized_name = path_dataset + \
			"/fragments/global_registration_optimized.json"
	run_posegraph_optimization(pose_graph_name, pose_graph_optmized_name,
			max_correspondence_distance = 0.075)
