# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

import sys
sys.path.append("../..")
from py3d import *
import numpy as np

if __name__ == "__main__":
	print("Testing PoseGraph in py3d ...")
	pose_graph = ReadPoseGraph("../TestData/test_pose_graph.json")
	print(pose_graph)
	WritePoseGraph("../TestData/test_pose_graph_copy.json", pose_graph)
	print("")
