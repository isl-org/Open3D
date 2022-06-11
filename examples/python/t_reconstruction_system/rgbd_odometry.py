import numpy as np
import open3d as o3d
from config import ConfigParser
from common import load_rgbd_file_names, load_depth_file_names, save_poses, load_intrinsic, load_extrinsics


def read_rgbd_image(color_file, depth_file, convert_rgb_to_intensity):
    color = o3d.io.read_image(color_file)
    depth = o3d.io.read_image(depth_file)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color,
        depth,
        depth_scale=1000.0,
        depth_trunc=3.0,
        convert_rgb_to_intensity=convert_rgb_to_intensity)
    return rgbd_image


if __name__ == '__main__':
    parser = ConfigParser()
    parser.add(
        '--config',
        is_config_file=True,
        help='YAML config file path. Please refer to default_config.yml as a '
        'reference. It overrides the default config file, but will be '
        'overridden by other command line inputs.')
    config = parser.get_config()

    depth_file_names, color_file_names = load_rgbd_file_names(config)

    intrinsic = load_intrinsic(config)

    i = 0
    j = 1

    depth_src = o3d.t.io.read_image(depth_file_names[i])
    color_src = o3d.t.io.read_image(color_file_names[i])

    depth_dst = o3d.t.io.read_image(depth_file_names[j])
    color_dst = o3d.t.io.read_image(color_file_names[j])

    rgbd_src = o3d.t.geometry.RGBDImage(color_src, depth_src)
    rgbd_dst = o3d.t.geometry.RGBDImage(color_dst, depth_dst)

    rgbd_src_legacy = read_rgbd_image(color_file_names[i], depth_file_names[i],
                                      True)
    rgbd_dst_legacy = read_rgbd_image(color_file_names[j], depth_file_names[j],
                                      True)
    intrinsic_legacy = o3d.camera.PinholeCameraIntrinsic(
        o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
    success, trans, info = o3d.pipelines.odometry.compute_rgbd_odometry(
        rgbd_src_legacy, rgbd_dst_legacy, intrinsic_legacy, np.eye(4),
        o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm())
    print(info)

    res = o3d.t.pipelines.odometry.rgbd_odometry_multi_scale(
        rgbd_src, rgbd_dst, intrinsic)
    depth_srcf = depth_src.clip_transform(1000.0, 0.0, 3.0, np.nan)
    depth_dstf = depth_dst.clip_transform(1000.0, 0.0, 3.0, np.nan)
    info = o3d.t.pipelines.odometry.compute_odometry_information_matrix(
        depth_srcf.as_tensor(), depth_dstf.as_tensor(), intrinsic,
        res.transformation, 0.03)
    print(info)

    pcd_src = o3d.t.geometry.PointCloud.create_from_rgbd_image(
        rgbd_src, intrinsic)
    pcd_dst = o3d.t.geometry.PointCloud.create_from_rgbd_image(
        rgbd_dst, intrinsic)

    o3d.visualization.draw([pcd_src, pcd_dst])
    o3d.visualization.draw([pcd_src.transform(res.transformation), pcd_dst])
