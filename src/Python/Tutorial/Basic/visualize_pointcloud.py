import sys
sys.path.append("../..")
from py3d import *

if __name__ == "__main__":
	pcd = ReadPointCloud("../../TestData/fragment.ply")
	DrawGeometries([pcd])
	pcd = ReadPointCloud("../../TestData/ICP/cloud_bin_0.pcd")
	DrawGeometries([pcd])
