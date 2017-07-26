import sys
sys.path.append("../..")
from py3d import *
from trajectory_io import *
import numpy as np

if __name__ == "__main__":
    intrinsic = PinholeCameraIntrinsic.PrimeSenseDefault
    extrinsic = read_trajectory("../../TestData/RGBD/odometry.log")
    volume = UniformTSDFVolume(length = 4.0, resolution = 512, sdf_trunc = 0.04,
            with_color = True)

    for i in range(len(extrinsic)):
        print("Integrate {:d}-th image into the volume.".format(i))
        color = ReadImage("../../TestData/RGBD/color/{:05d}.jpg".format(i))
        depth = ReadImage("../../TestData/RGBD/depth/{:05d}.png".format(i))
        rgbd = CreateRGBDImageFromColorAndDepth(color, depth, depth_trunc = 4.0,
                convert_rgb_to_intensity = False)
        volume.Integrate(rgbd, intrinsic, np.linalg.inv(extrinsic[i].pose))

    print("Extract a triangle mesh from the volume and visualize it.")
    mesh = volume.ExtractTriangleMesh()
    mesh.ComputeVertexNormals()
    DrawGeometries([mesh])
