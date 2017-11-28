import sys
sys.path.append("../..")
from py3d import *


def optimize_posegraph(pose_graph_name, pose_graph_optmized_name):
	# to display messages from global_optimization
	set_verbosity_level(VerbosityLevel.Debug)
	method = GlobalOptimizationLevenbergMarquardt()
	criteria = GlobalOptimizationConvergenceCriteria()
	line_process_option = GlobalOptimizationLineProcessOption()
	pose_graph = read_pose_graph(pose_graph_name)
	global_optimization(pose_graph, method, criteria, line_process_option)
	write_pose_graph(pose_graph_optmized_name, pose_graph)
	set_verbosity_level(VerbosityLevel.Error)
