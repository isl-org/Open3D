import sys
sys.path.append("../..")
from py3d import *
from trajectory_io import *
import numpy as np

if __name__ == "__main__":
    intrinsic = PinholeCameraIntrinsic.PrimeSenseDefault
    camera_poses = read_trajectory("../../TestData/RGBD/odometry.log")
    #volume = UniformTSDFVolume(length = 4.0, resolution = 512, sdf_trunc = 0.04,
    #        with_color = True)
    volume = ScalableTSDFVolume(voxel_length = 4.0 / 512.0, sdf_trunc = 0.04,\
            with_color = True)

    for i in range(len(camera_poses)):
        print("Integrate {:d}-th image into the volume.".format(i))
        color = read_image("../../TestData/RGBD/color/{:05d}.jpg".format(i))
        depth = read_image("../../TestData/RGBD/depth/{:05d}.png".format(i))
        rgbd = create_rgbd_image_from_color_and_depth(color, depth, depth_trunc = 4.0,
                convert_rgb_to_intensity = False)
        volume.Integrate(rgbd, intrinsic, np.linalg.inv(camera_poses[i].pose))

    print("Extract a triangle mesh from the volume and visualize it.")
    mesh = volume.ExtractTriangleMesh()
    mesh.compute_vertex_normals()
    DrawGeometries([mesh])
