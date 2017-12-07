import sys
sys.path.append("../..")
from py3d import *

if __name__ == "__main__":

	print("Testing PoseGraph in py3d ...")
	pose_graph = read_pose_graph("../../TestData/test_pose_graph.json")
	print(pose_graph)
	write_pose_graph("../../TestData/test_pose_graph_copy.json", pose_graph)
	print("")
