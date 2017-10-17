import sys
sys.path.append("../..")
from py3d import *


def optimize_posegraph(pose_graph_name, pose_graph_optmized_name):
	# to display messages from GlobalOptimization
	SetVerbosityLevel(VerbosityLevel.Debug)
	method = GlobalOptimizationLevenbergMarquardt()
	criteria = GlobalOptimizationConvergenceCriteria()
	option = GlobalOptimizationOption(
			max_correspondence_distance = 0.03,
			edge_prune_threshold = 0.25,
			unchanged_node = 0)
	pose_graph = ReadPoseGraph(pose_graph_name)
	GlobalOptimization(pose_graph, method, criteria, option)
	WritePoseGraph(pose_graph_optmized_name, pose_graph)
	SetVerbosityLevel(VerbosityLevel.Error)
