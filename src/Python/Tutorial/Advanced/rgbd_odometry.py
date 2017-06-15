import sys
sys.path.append("../..")
from py3d import *

if __name__ == "__main__":
	image0 = ReadImage("../../TestData/RGBD/color/00000.jpg")
	depth0 = ReadImage("../../TestData/RGBD/depth/00000.png")
	image1 = ReadImage("../../TestData/RGBD/color/00001.jpg")
	depth1 = ReadImage("../../TestData/RGBD/depth/00001.png")
	option = OdometryOption();
	option.intrinsic_path = "../../TestData/camera.json"
	print(option)
	[success, trans, info] = ComputeRGBDOdometry(
			image0, depth0, image1, depth1, option)
	if success:
		print(trans)
