sys.path.append("../..")
from py3d import *


def optimize_posegraph(pose_graph_name, pose_graph_optmized_name):
	# to display messages from GlobalOptimization
	SetVerbosityLevel(VerbosityLevel.Debug)
	method = GlobalOptimizationLevenbergMarquardt()
	criteria = GlobalOptimizationConvergenceCriteria()
	line_process_option = GlobalOptimizationLineProcessOption()
	pose_graph = ReadPoseGraph(pose_graph_name)
	GlobalOptimization(pose_graph, method, criteria, line_process_option)
	WritePoseGraph(pose_graph_optmized_name, pose_graph)
	SetVerbosityLevel(VerbosityLevel.Error)
