# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# The MIT License (MIT)
#
# Copyright (c) 2018-2021 www.open3d.org
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
# ----------------------------------------------------------------------------

import open3d as o3d

from pathlib import Path
import shutil


def test_dataset_base():
    default_data_root = Path.home() / "open3d_data"

    ds = o3d.data.Dataset("some_prefix")
    assert Path(ds.data_root) == default_data_root

    ds_custom = o3d.data.Dataset("some_prefix", "/my/custom/data_root")
    assert Path(ds_custom.data_root) == Path("/my/custom/data_root")
    assert ds_custom.prefix == "some_prefix"
    assert ds_custom.download_dir == "/my/custom/data_root/download/some_prefix"
    assert ds_custom.extract_dir == "/my/custom/data_root/extract/some_prefix"


def get_test_data_dirs(prefix):
    gt_data_root = Path.home() / "open3d_data"
    gt_download_dir = gt_data_root / "download" / prefix
    gt_extract_dir = gt_data_root / "extract" / prefix
    return gt_data_root, gt_download_dir, gt_extract_dir


def test_simple_dataset_base():
    gt_prefix = "BunnyMesh"
    gt_data_root, gt_download_dir, gt_extract_dir = get_test_data_dirs(
        gt_prefix)

    url_mirrors = [
        "https://github.com/isl-org/open3d_downloads/releases/download/"
        "20220201-data/BunnyMesh.ply"
    ]
    md5 = "568f871d1a221ba6627569f1e6f9a3f2"

    single_download_dataset = o3d.data.SingleDownloadDataset(
        gt_prefix,
        url_mirrors,
        md5,
        True,
    )

    assert Path(gt_extract_dir / "BunnyMesh.ply").is_file()
    assert Path(gt_download_dir / "BunnyMesh.ply").is_file()

    assert single_download_dataset.prefix == gt_prefix
    assert Path(single_download_dataset.data_root) == gt_data_root
    assert Path(single_download_dataset.download_dir) == gt_download_dir
    assert Path(single_download_dataset.extract_dir) == gt_extract_dir


def test_demo_icp_pointclouds():
    gt_prefix = "DemoICPPointClouds"
    gt_data_root, gt_download_dir, gt_extract_dir = get_test_data_dirs(
        gt_prefix)

    demo_icp = o3d.data.DemoICPPointClouds()
    assert Path(gt_download_dir).is_dir()

    gt_paths = [
        gt_extract_dir / "cloud_bin_0.pcd", gt_extract_dir / "cloud_bin_1.pcd",
        gt_extract_dir / "cloud_bin_2.pcd"
    ]

    assert len(demo_icp.paths) == len(gt_paths)
    for gt_path, demo_icp_path in zip(gt_paths, demo_icp.paths):
        assert Path(gt_path) == Path(demo_icp_path)
        assert Path(gt_path).is_file()

    assert Path(demo_icp.transformation_log_path) == Path(gt_extract_dir /
                                                          "init.log")
    assert Path(demo_icp.transformation_log_path).is_file()

    assert demo_icp.prefix == gt_prefix
    assert Path(demo_icp.data_root) == gt_data_root
    assert Path(demo_icp.download_dir) == gt_download_dir
    assert Path(demo_icp.extract_dir) == gt_extract_dir


def test_demo_colored_icp_pointclouds():
    gt_prefix = "DemoColoredICPPointClouds"
    gt_data_root, gt_download_dir, gt_extract_dir = get_test_data_dirs(
        gt_prefix)

    demo_colored_icp = o3d.data.DemoColoredICPPointClouds()
    assert Path(gt_download_dir).is_dir()

    gt_paths = [
        gt_extract_dir / "frag_115.ply", gt_extract_dir / "frag_116.ply"
    ]

    assert len(demo_colored_icp.paths) == len(gt_paths)
    for gt_path, demo_colored_icp_path in zip(gt_paths, demo_colored_icp.paths):
        assert Path(gt_path) == Path(demo_colored_icp_path)
        assert Path(gt_path).is_file()

    assert demo_colored_icp.prefix == gt_prefix
    assert Path(demo_colored_icp.data_root) == gt_data_root
    assert Path(demo_colored_icp.download_dir) == gt_download_dir
    assert Path(demo_colored_icp.extract_dir) == gt_extract_dir


def test_demo_crop_pointcloud():
    gt_prefix = "DemoCropPointCloud"
    gt_data_root, gt_download_dir, gt_extract_dir = get_test_data_dirs(
        gt_prefix)

    demo_crop_pcd = o3d.data.DemoCropPointCloud()
    assert Path(gt_download_dir).is_dir()

    assert Path(demo_crop_pcd.point_cloud_path) == gt_extract_dir / \
        "fragment.ply"
    assert Path(demo_crop_pcd.point_cloud_path).is_file()
    assert Path(demo_crop_pcd.cropped_json_path) == gt_extract_dir / \
        "cropped.json"
    assert Path(demo_crop_pcd.point_cloud_path).is_file()

    assert demo_crop_pcd.prefix == gt_prefix
    assert Path(demo_crop_pcd.data_root) == gt_data_root
    assert Path(demo_crop_pcd.download_dir) == gt_download_dir
    assert Path(demo_crop_pcd.extract_dir) == gt_extract_dir


def test_demo_feature_matching_point_clouds():
    gt_prefix = "DemoFeatureMatchingPointClouds"
    gt_data_root, gt_download_dir, gt_extract_dir = get_test_data_dirs(
        gt_prefix)

    demo_feature_matching = o3d.data.DemoFeatureMatchingPointClouds()
    assert Path(gt_download_dir).is_dir()

    gt_point_cloud_paths = [
        gt_extract_dir / "cloud_bin_0.pcd", gt_extract_dir / "cloud_bin_1.pcd"
    ]
    assert len(
        demo_feature_matching.point_cloud_paths) == len(gt_point_cloud_paths)
    for gt_path, demo_feature_matching_point_cloud_path in zip(
            gt_point_cloud_paths, demo_feature_matching.point_cloud_paths):
        assert Path(demo_feature_matching_point_cloud_path) == gt_path
        assert Path(gt_path).is_file()

    gt_fpfh_feature_paths = [
        gt_extract_dir / "cloud_bin_0.fpfh.bin",
        gt_extract_dir / "cloud_bin_1.fpfh.bin"
    ]
    assert len(
        demo_feature_matching.fpfh_feature_paths) == len(gt_fpfh_feature_paths)
    for gt_path, demo_feature_matching_fpfh_feature_path in zip(
            gt_fpfh_feature_paths, demo_feature_matching.fpfh_feature_paths):
        assert Path(demo_feature_matching_fpfh_feature_path) == gt_path
        assert Path(gt_path).is_file()

    gt_l32d_feature_paths = [
        gt_extract_dir / "cloud_bin_0.d32.bin",
        gt_extract_dir / "cloud_bin_1.d32.bin"
    ]
    assert len(
        demo_feature_matching.l32d_feature_paths) == len(gt_l32d_feature_paths)
    for gt_path, demo_feature_matching_l32d_feature_path in zip(
            gt_l32d_feature_paths, demo_feature_matching.l32d_feature_paths):
        assert Path(demo_feature_matching_l32d_feature_path) == gt_path
        assert Path(gt_path).is_file()

    assert demo_feature_matching.prefix == gt_prefix
    assert Path(demo_feature_matching.data_root) == gt_data_root
    assert Path(demo_feature_matching.download_dir) == gt_download_dir
    assert Path(demo_feature_matching.extract_dir) == gt_extract_dir


def test_demo_pose_graph_optimization():
    gt_prefix = "DemoPoseGraphOptimization"
    gt_data_root, gt_download_dir, gt_extract_dir = get_test_data_dirs(
        gt_prefix)

    demo_pose_optimization = o3d.data.DemoPoseGraphOptimization()
    assert Path(gt_download_dir).is_dir()

    assert Path(demo_pose_optimization.pose_graph_fragment_path) == gt_extract_dir / \
        "pose_graph_example_fragment.json"
    assert Path(demo_pose_optimization.pose_graph_fragment_path).is_file()
    assert Path(demo_pose_optimization.pose_graph_global_path) == gt_extract_dir / \
        "pose_graph_example_global.json"
    assert Path(demo_pose_optimization.pose_graph_global_path).is_file()

    assert demo_pose_optimization.prefix == gt_prefix
    assert Path(demo_pose_optimization.data_root) == gt_data_root
    assert Path(demo_pose_optimization.download_dir) == gt_download_dir
    assert Path(demo_pose_optimization.extract_dir) == gt_extract_dir


def test_pcd_point_cloud():
    gt_prefix = "PCDPointCloud"
    gt_data_root, gt_download_dir, gt_extract_dir = get_test_data_dirs(
        gt_prefix)

    pcd_pointcloud = o3d.data.PCDPointCloud()
    assert Path(gt_download_dir).is_dir()

    assert Path(pcd_pointcloud.path) == gt_extract_dir / "fragment.pcd"
    assert Path(pcd_pointcloud.path).is_file()

    assert pcd_pointcloud.prefix == gt_prefix
    assert Path(pcd_pointcloud.data_root) == gt_data_root
    assert Path(pcd_pointcloud.download_dir) == gt_download_dir
    assert Path(pcd_pointcloud.extract_dir) == gt_extract_dir


def test_ply_point_cloud():
    gt_prefix = "PLYPointCloud"
    gt_data_root, gt_download_dir, gt_extract_dir = get_test_data_dirs(
        gt_prefix)

    ply_pointcloud = o3d.data.PLYPointCloud()
    assert Path(gt_download_dir).is_dir()

    assert Path(ply_pointcloud.path) == gt_extract_dir / "fragment.ply"
    assert Path(ply_pointcloud.path).is_file()

    assert ply_pointcloud.prefix == gt_prefix
    assert Path(ply_pointcloud.data_root) == gt_data_root
    assert Path(ply_pointcloud.download_dir) == gt_download_dir
    assert Path(ply_pointcloud.extract_dir) == gt_extract_dir


def test_sample_nyu_rgbd_image():
    gt_prefix = "SampleNYURGBDImage"
    gt_data_root, gt_download_dir, gt_extract_dir = get_test_data_dirs(
        gt_prefix)

    rgbd_image_nyu = o3d.data.SampleNYURGBDImage()
    assert Path(gt_download_dir).is_dir()

    assert Path(rgbd_image_nyu.color_path) == gt_extract_dir / "NYU_color.ppm"
    assert Path(rgbd_image_nyu.color_path).is_file()

    assert Path(rgbd_image_nyu.depth_path) == gt_extract_dir / "NYU_depth.pgm"
    assert Path(rgbd_image_nyu.depth_path).is_file()

    assert rgbd_image_nyu.prefix == gt_prefix
    assert Path(rgbd_image_nyu.data_root) == gt_data_root
    assert Path(rgbd_image_nyu.download_dir) == gt_download_dir
    assert Path(rgbd_image_nyu.extract_dir) == gt_extract_dir


def test_sample_sun_rgbd_image():
    gt_prefix = "SampleSUNRGBDImage"
    gt_data_root, gt_download_dir, gt_extract_dir = get_test_data_dirs(
        gt_prefix)

    rgbd_image_sun = o3d.data.SampleSUNRGBDImage()
    assert Path(gt_download_dir).is_dir()

    assert Path(rgbd_image_sun.color_path) == gt_extract_dir / "SUN_color.jpg"
    assert Path(rgbd_image_sun.color_path).is_file()

    assert Path(rgbd_image_sun.depth_path) == gt_extract_dir / "SUN_depth.png"
    assert Path(rgbd_image_sun.depth_path).is_file()

    assert rgbd_image_sun.prefix == gt_prefix
    assert Path(rgbd_image_sun.data_root) == gt_data_root
    assert Path(rgbd_image_sun.download_dir) == gt_download_dir
    assert Path(rgbd_image_sun.extract_dir) == gt_extract_dir


def test_sample_tum_rgbd_image():
    gt_prefix = "SampleTUMRGBDImage"
    gt_data_root, gt_download_dir, gt_extract_dir = get_test_data_dirs(
        gt_prefix)

    rgbd_image_tum = o3d.data.SampleTUMRGBDImage()
    assert Path(gt_download_dir).is_dir()

    assert Path(rgbd_image_tum.color_path) == gt_extract_dir / "TUM_color.png"
    assert Path(rgbd_image_tum.color_path).is_file()

    assert Path(rgbd_image_tum.depth_path) == gt_extract_dir / "TUM_depth.png"
    assert Path(rgbd_image_tum.depth_path).is_file()

    assert rgbd_image_tum.prefix == gt_prefix
    assert Path(rgbd_image_tum.data_root) == gt_data_root
    assert Path(rgbd_image_tum.download_dir) == gt_download_dir
    assert Path(rgbd_image_tum.extract_dir) == gt_extract_dir


def test_sample_redwood_rgbd_images():
    gt_prefix = "SampleRedwoodRGBDImages"
    gt_data_root, gt_download_dir, gt_extract_dir = get_test_data_dirs(
        gt_prefix)

    rgbd_dataset_redwood = o3d.data.SampleRedwoodRGBDImages()
    assert Path(gt_download_dir).is_dir()

    gt_color_paths = [
        gt_extract_dir / "color" / "00000.jpg",
        gt_extract_dir / "color" / "00001.jpg",
        gt_extract_dir / "color" / "00002.jpg",
        gt_extract_dir / "color" / "00003.jpg",
        gt_extract_dir / "color" / "00004.jpg"
    ]
    assert len(rgbd_dataset_redwood.color_paths) == len(gt_color_paths)
    for gt_path, rgbd_dataset_redwood_color_path in zip(
            gt_color_paths, rgbd_dataset_redwood.color_paths):
        assert Path(rgbd_dataset_redwood_color_path) == gt_path
        assert Path(gt_path).is_file()

    gt_depth_paths = [
        gt_extract_dir / "depth" / "00000.png",
        gt_extract_dir / "depth" / "00001.png",
        gt_extract_dir / "depth" / "00002.png",
        gt_extract_dir / "depth" / "00003.png",
        gt_extract_dir / "depth" / "00004.png"
    ]
    assert len(rgbd_dataset_redwood.depth_paths) == len(gt_depth_paths)
    for gt_path, rgbd_dataset_redwood_depth_path in zip(
            gt_depth_paths, rgbd_dataset_redwood.depth_paths):
        assert Path(rgbd_dataset_redwood_depth_path) == gt_path
        assert Path(gt_path).is_file()

    assert Path(rgbd_dataset_redwood.trajectory_log_path
               ) == gt_extract_dir / "trajectory.log"
    assert Path(rgbd_dataset_redwood.odometry_log_path
               ) == gt_extract_dir / "odometry.log"
    assert Path(
        rgbd_dataset_redwood.rgbd_match_path) == gt_extract_dir / "rgbd.match"
    assert Path(rgbd_dataset_redwood.reconstruction_path
               ) == gt_extract_dir / "example_tsdf_pcd.ply"

    assert rgbd_dataset_redwood.prefix == gt_prefix
    assert Path(rgbd_dataset_redwood.data_root) == gt_data_root
    assert Path(rgbd_dataset_redwood.download_dir) == gt_download_dir
    assert Path(rgbd_dataset_redwood.extract_dir) == gt_extract_dir


def test_sample_fountain_rgbd_images():
    gt_prefix = "SampleFountainRGBDImages"
    gt_data_root, gt_download_dir, gt_extract_dir = get_test_data_dirs(
        gt_prefix)

    fountain_dataset = o3d.data.SampleFountainRGBDImages()
    assert Path(gt_download_dir).is_dir()

    gt_color_paths = [
        gt_extract_dir / "image" / "0000010-000001228920.jpg",
        gt_extract_dir / "image" / "0000031-000004096400.jpg",
        gt_extract_dir / "image" / "0000044-000005871507.jpg",
        gt_extract_dir / "image" / "0000064-000008602440.jpg",
        gt_extract_dir / "image" / "0000110-000014883587.jpg",
        gt_extract_dir / "image" / "0000156-000021164733.jpg",
        gt_extract_dir / "image" / "0000200-000027172787.jpg",
        gt_extract_dir / "image" / "0000215-000029220987.jpg",
        gt_extract_dir / "image" / "0000255-000034682853.jpg",
        gt_extract_dir / "image" / "0000299-000040690907.jpg",
        gt_extract_dir / "image" / "0000331-000045060400.jpg",
        gt_extract_dir / "image" / "0000368-000050112627.jpg",
        gt_extract_dir / "image" / "0000412-000056120680.jpg",
        gt_extract_dir / "image" / "0000429-000058441973.jpg",
        gt_extract_dir / "image" / "0000474-000064586573.jpg",
        gt_extract_dir / "image" / "0000487-000066361680.jpg",
        gt_extract_dir / "image" / "0000526-000071687000.jpg",
        gt_extract_dir / "image" / "0000549-000074827573.jpg",
        gt_extract_dir / "image" / "0000582-000079333613.jpg",
        gt_extract_dir / "image" / "0000630-000085887853.jpg",
        gt_extract_dir / "image" / "0000655-000089301520.jpg",
        gt_extract_dir / "image" / "0000703-000095855760.jpg",
        gt_extract_dir / "image" / "0000722-000098450147.jpg",
        gt_extract_dir / "image" / "0000771-000105140933.jpg",
        gt_extract_dir / "image" / "0000792-000108008413.jpg",
        gt_extract_dir / "image" / "0000818-000111558627.jpg",
        gt_extract_dir / "image" / "0000849-000115791573.jpg",
        gt_extract_dir / "image" / "0000883-000120434160.jpg",
        gt_extract_dir / "image" / "0000896-000122209267.jpg",
        gt_extract_dir / "image" / "0000935-000127534587.jpg",
        gt_extract_dir / "image" / "0000985-000134361920.jpg",
        gt_extract_dir / "image" / "0001028-000140233427.jpg",
        gt_extract_dir / "image" / "0001061-000144739467.jpg"
    ]
    assert len(fountain_dataset.color_paths) == len(gt_color_paths)
    for gt_path, fountain_dataset_color_path in zip(
            gt_color_paths, fountain_dataset.color_paths):
        assert Path(fountain_dataset_color_path) == gt_path
        assert Path(gt_path).is_file()

    gt_depth_paths = [
        gt_extract_dir / "depth" / "0000038-000001234662.png",
        gt_extract_dir / "depth" / "0000124-000004104418.png",
        gt_extract_dir / "depth" / "0000177-000005872988.png",
        gt_extract_dir / "depth" / "0000259-000008609267.png",
        gt_extract_dir / "depth" / "0000447-000014882686.png",
        gt_extract_dir / "depth" / "0000635-000021156105.png",
        gt_extract_dir / "depth" / "0000815-000027162570.png",
        gt_extract_dir / "depth" / "0000877-000029231463.png",
        gt_extract_dir / "depth" / "0001040-000034670651.png",
        gt_extract_dir / "depth" / "0001220-000040677116.png",
        gt_extract_dir / "depth" / "0001351-000045048488.png",
        gt_extract_dir / "depth" / "0001503-000050120614.png",
        gt_extract_dir / "depth" / "0001683-000056127079.png",
        gt_extract_dir / "depth" / "0001752-000058429557.png",
        gt_extract_dir / "depth" / "0001937-000064602868.png",
        gt_extract_dir / "depth" / "0001990-000066371438.png",
        gt_extract_dir / "depth" / "0002149-000071677149.png",
        gt_extract_dir / "depth" / "0002243-000074813859.png",
        gt_extract_dir / "depth" / "0002378-000079318707.png",
        gt_extract_dir / "depth" / "0002575-000085892450.png",
        gt_extract_dir / "depth" / "0002677-000089296113.png",
        gt_extract_dir / "depth" / "0002874-000095869855.png",
        gt_extract_dir / "depth" / "0002951-000098439288.png",
        gt_extract_dir / "depth" / "0003152-000105146507.png",
        gt_extract_dir / "depth" / "0003238-000108016262.png",
        gt_extract_dir / "depth" / "0003344-000111553403.png",
        gt_extract_dir / "depth" / "0003471-000115791298.png",
        gt_extract_dir / "depth" / "0003610-000120429623.png",
        gt_extract_dir / "depth" / "0003663-000122198194.png",
        gt_extract_dir / "depth" / "0003823-000127537274.png",
        gt_extract_dir / "depth" / "0004028-000134377970.png",
        gt_extract_dir / "depth" / "0004203-000140217589.png",
        gt_extract_dir / "depth" / "0004339-000144755807.png"
    ]
    assert len(fountain_dataset.depth_paths) == len(gt_depth_paths)
    for gt_path, fountain_dataset_depth_path in zip(
            gt_depth_paths, fountain_dataset.depth_paths):
        assert Path(fountain_dataset_depth_path) == gt_path
        assert Path(gt_path).is_file()

    assert Path(fountain_dataset.keyframe_poses_log_path
               ) == gt_extract_dir / "scene" / "key.log"
    assert Path(fountain_dataset.reconstruction_path
               ) == gt_extract_dir / "scene" / "integrated.ply"

    assert fountain_dataset.prefix == gt_prefix
    assert Path(fountain_dataset.data_root) == gt_data_root
    assert Path(fountain_dataset.download_dir) == gt_download_dir
    assert Path(fountain_dataset.extract_dir) == gt_extract_dir


def test_eagle():
    gt_prefix = "EaglePointCloud"
    gt_data_root, gt_download_dir, gt_extract_dir = get_test_data_dirs(
        gt_prefix)

    eagle = o3d.data.EaglePointCloud()
    assert Path(gt_download_dir).is_dir()

    assert Path(eagle.path) == gt_extract_dir / "EaglePointCloud.ply"
    assert Path(eagle.path).is_file()

    assert eagle.prefix == gt_prefix
    assert Path(eagle.data_root) == gt_data_root
    assert Path(eagle.download_dir) == gt_download_dir
    assert Path(eagle.extract_dir) == gt_extract_dir


def test_armadillo():
    gt_prefix = "ArmadilloMesh"
    gt_data_root, gt_download_dir, gt_extract_dir = get_test_data_dirs(
        gt_prefix)

    armadillo = o3d.data.ArmadilloMesh()
    assert Path(gt_download_dir).is_dir()

    assert Path(armadillo.path) == gt_extract_dir / "ArmadilloMesh.ply"
    assert Path(armadillo.path).is_file()

    assert armadillo.prefix == gt_prefix
    assert Path(armadillo.data_root) == gt_data_root
    assert Path(armadillo.download_dir) == gt_download_dir
    assert Path(armadillo.extract_dir) == gt_extract_dir


def test_bunny():
    gt_prefix = "BunnyMesh"
    gt_data_root, gt_download_dir, gt_extract_dir = get_test_data_dirs(
        gt_prefix)

    bunny = o3d.data.BunnyMesh()
    assert Path(gt_download_dir).is_dir()

    assert Path(bunny.path) == gt_extract_dir / "BunnyMesh.ply"
    assert Path(bunny.path).is_file()

    assert bunny.prefix == gt_prefix
    assert Path(bunny.data_root) == gt_data_root
    assert Path(bunny.download_dir) == gt_download_dir
    assert Path(bunny.extract_dir) == gt_extract_dir


def test_knot():
    gt_prefix = "KnotMesh"
    gt_data_root, gt_download_dir, gt_extract_dir = get_test_data_dirs(
        gt_prefix)

    knot = o3d.data.KnotMesh()
    assert Path(gt_download_dir).is_dir()

    assert Path(knot.path) == gt_extract_dir / "KnotMesh.ply"
    assert Path(knot.path).is_file()

    assert knot.prefix == gt_prefix
    assert Path(knot.data_root) == gt_data_root
    assert Path(knot.download_dir) == gt_download_dir
    assert Path(knot.extract_dir) == gt_extract_dir


def test_monkey():
    gt_prefix = "MonkeyModel"
    gt_data_root, gt_download_dir, gt_extract_dir = get_test_data_dirs(
        gt_prefix)

    monkey = o3d.data.MonkeyModel()
    assert Path(gt_download_dir).is_dir()

    gt_path_map = {
        "albedo": Path(gt_extract_dir) / "albedo.png",
        "ao": Path(gt_extract_dir) / "ao.png",
        "metallic": Path(gt_extract_dir) / "metallic.png",
        "monkey_material": Path(gt_extract_dir) / "monkey.mtl",
        "monkey_model": Path(gt_extract_dir) / "monkey.obj",
        "monkey_solid_material": Path(gt_extract_dir) / "monkey_solid.mtl",
        "monkey_solid_model": Path(gt_extract_dir) / "monkey_solid.obj",
        "normal": Path(gt_extract_dir) / "normal.png",
        "roughness": Path(gt_extract_dir) / "roughness.png"
    }

    for file_name in monkey.path_map:
        assert Path(monkey.path_map[file_name]) == gt_path_map[file_name]
        assert Path(monkey.path_map[file_name]).is_file()

    assert Path(monkey.path) == gt_extract_dir / "monkey.obj"
    assert Path(monkey.path).is_file()

    assert monkey.prefix == gt_prefix
    assert Path(monkey.data_root) == gt_data_root
    assert Path(monkey.download_dir) == gt_download_dir
    assert Path(monkey.extract_dir) == gt_extract_dir


def test_sword():
    gt_prefix = "SwordModel"
    gt_data_root, gt_download_dir, gt_extract_dir = get_test_data_dirs(
        gt_prefix)

    sword = o3d.data.SwordModel()
    assert Path(gt_download_dir).is_dir()

    gt_path_map = {
        "sword_material": Path(gt_extract_dir) / "UV.mtl",
        "sword_model": Path(gt_extract_dir) / "UV.obj",
        "base_color": Path(gt_extract_dir) / "UV_blinn1SG_BaseColor.png",
        "metallic": Path(gt_extract_dir) / "UV_blinn1SG_Metallic.png",
        "normal": Path(gt_extract_dir) / "UV_blinn1SG_Normal.png",
        "roughness": Path(gt_extract_dir) / "UV_blinn1SG_Roughness.png"
    }

    for file_name in sword.path_map:
        assert Path(sword.path_map[file_name]) == gt_path_map[file_name]
        assert Path(sword.path_map[file_name]).is_file()

    assert Path(sword.path) == gt_extract_dir / "UV.obj"
    assert Path(sword.path).is_file()

    assert sword.prefix == gt_prefix
    assert Path(sword.data_root) == gt_data_root
    assert Path(sword.download_dir) == gt_download_dir
    assert Path(sword.extract_dir) == gt_extract_dir


def test_crate():
    gt_prefix = "CrateModel"
    gt_data_root, gt_download_dir, gt_extract_dir = get_test_data_dirs(
        gt_prefix)

    crate = o3d.data.CrateModel()
    assert Path(gt_download_dir).is_dir()

    gt_path_map = {
        "crate_material": Path(gt_extract_dir) / "crate.mtl",
        "crate_model": Path(gt_extract_dir) / "crate.obj",
        "texture_image": Path(gt_extract_dir) / "crate.jpg"
    }

    for file_name in crate.path_map:
        assert Path(crate.path_map[file_name]) == gt_path_map[file_name]
        assert Path(crate.path_map[file_name]).is_file()

    assert Path(crate.path) == gt_extract_dir / "crate.obj"
    assert Path(crate.path).is_file()

    assert crate.prefix == gt_prefix
    assert Path(crate.data_root) == gt_data_root
    assert Path(crate.download_dir) == gt_download_dir
    assert Path(crate.extract_dir) == gt_extract_dir


def test_flight_helmet():
    gt_prefix = "FlightHelmetModel"
    gt_data_root, gt_download_dir, gt_extract_dir = get_test_data_dirs(
        gt_prefix)

    helmet = o3d.data.FlightHelmetModel()
    assert Path(gt_download_dir).is_dir()

    gt_path_map = {
        "flight_helmet":
            Path(gt_extract_dir) / "FlightHelmet.gltf",
        "flight_helmet_bin":
            Path(gt_extract_dir) / "FlightHelmet.bin",
        "mat_glass_plastic_base":
            Path(gt_extract_dir) /
            "FlightHelmet_Materials_GlassPlasticMat_BaseColor.png",
        "mat_glass_plastic_normal":
            Path(gt_extract_dir) /
            "FlightHelmet_Materials_GlassPlasticMat_Normal.png",
        "mat_glass_plastic_occlusion_rough_metal":
            Path(gt_extract_dir) / "FlightHelmet_Materials_GlassPlasticMat_"
            "OcclusionRoughMetal.png",
        "mat_leather_parts_base":
            Path(gt_extract_dir) /
            "FlightHelmet_Materials_LeatherPartsMat_BaseColor.png",
        "mat_leather_parts_normal":
            Path(gt_extract_dir) /
            "FlightHelmet_Materials_LeatherPartsMat_Normal.png",
        "mat_leather_parts_occlusion_rough_metal":
            Path(gt_extract_dir) / "FlightHelmet_Materials_LeatherPartsMat_"
            "OcclusionRoughMetal.png",
        "mat_lenses_base":
            Path(gt_extract_dir) /
            "FlightHelmet_Materials_LensesMat_BaseColor.png",
        "mat_lenses_normal":
            Path(gt_extract_dir) /
            "FlightHelmet_Materials_LensesMat_Normal.png",
        "mat_lenses_occlusion_rough_metal":
            Path(gt_extract_dir) / "FlightHelmet_Materials_LensesMat_"
            "OcclusionRoughMetal.png",
        "mat_metal_parts_base":
            Path(gt_extract_dir) /
            "FlightHelmet_Materials_MetalPartsMat_BaseColor.png",
        "mat_metal_parts_normal":
            Path(gt_extract_dir) /
            "FlightHelmet_Materials_MetalPartsMat_Normal.png",
        "mat_metal_parts_occlusion_rough_metal":
            Path(gt_extract_dir) / "FlightHelmet_Materials_MetalPartsMat_"
            "OcclusionRoughMetal.png",
        "mat_rubber_wood_base":
            Path(gt_extract_dir) /
            "FlightHelmet_Materials_RubberWoodMat_BaseColor.png",
        "mat_rubber_wood_normal":
            Path(gt_extract_dir) /
            "FlightHelmet_Materials_RubberWoodMat_Normal.png",
        "mat_rubber_wood_occlusion_rough_metal":
            Path(gt_extract_dir) / "FlightHelmet_Materials_RubberWoodMat_"
            "OcclusionRoughMetal.png"
    }

    for file_name in helmet.path_map:
        assert Path(helmet.path_map[file_name]) == gt_path_map[file_name]
        assert Path(helmet.path_map[file_name]).is_file()

    assert Path(helmet.path) == gt_extract_dir / "FlightHelmet.gltf"
    assert Path(helmet.path).is_file()

    assert helmet.prefix == gt_prefix
    assert Path(helmet.data_root) == gt_data_root
    assert Path(helmet.download_dir) == gt_download_dir
    assert Path(helmet.extract_dir) == gt_extract_dir


def test_metal_texture():
    gt_prefix = "MetalTexture"
    gt_data_root, gt_download_dir, gt_extract_dir = get_test_data_dirs(
        gt_prefix)

    data = o3d.data.MetalTexture()
    assert Path(gt_download_dir).is_dir()

    gt_path_map = {
        "albedo": Path(gt_extract_dir) / "Metal008_Color.jpg",
        "normal": Path(gt_extract_dir) / "Metal008_NormalDX.jpg",
        "roughness": Path(gt_extract_dir) / "Metal008_Roughness.jpg",
        "metallic": Path(gt_extract_dir) / "Metal008_Metalness.jpg"
    }

    for file_name in data.path_map:
        assert Path(data.path_map[file_name]) == gt_path_map[file_name]
        assert Path(data.path_map[file_name]).is_file()

    assert Path(data.albedo_texture_path) == gt_path_map["albedo"]
    assert Path(data.normal_texture_path) == gt_path_map["normal"]
    assert Path(data.roughness_texture_path) == gt_path_map["roughness"]
    assert Path(data.metallic_texture_path) == gt_path_map["metallic"]

    assert data.prefix == gt_prefix
    assert Path(data.data_root) == gt_data_root
    assert Path(data.download_dir) == gt_download_dir
    assert Path(data.extract_dir) == gt_extract_dir


def test_painted_plaster_texture():
    gt_prefix = "PaintedPlasterTexture"
    gt_data_root, gt_download_dir, gt_extract_dir = get_test_data_dirs(
        gt_prefix)

    data = o3d.data.PaintedPlasterTexture()
    assert Path(gt_download_dir).is_dir()

    gt_path_map = {
        "albedo": Path(gt_extract_dir) / "PaintedPlaster017_Color.jpg",
        "normal": Path(gt_extract_dir) / "PaintedPlaster017_NormalDX.jpg",
        "roughness": Path(gt_extract_dir) / "noiseTexture.png"
    }

    for file_name in data.path_map:
        assert Path(data.path_map[file_name]) == gt_path_map[file_name]
        assert Path(data.path_map[file_name]).is_file()

    assert Path(data.albedo_texture_path) == gt_path_map["albedo"]
    assert Path(data.normal_texture_path) == gt_path_map["normal"]
    assert Path(data.roughness_texture_path) == gt_path_map["roughness"]

    assert data.prefix == gt_prefix
    assert Path(data.data_root) == gt_data_root
    assert Path(data.download_dir) == gt_download_dir
    assert Path(data.extract_dir) == gt_extract_dir


def test_tiles_texture():
    gt_prefix = "TilesTexture"
    gt_data_root, gt_download_dir, gt_extract_dir = get_test_data_dirs(
        gt_prefix)

    data = o3d.data.TilesTexture()
    assert Path(gt_download_dir).is_dir()

    gt_path_map = {
        "albedo": Path(gt_extract_dir) / "Tiles074_Color.jpg",
        "normal": Path(gt_extract_dir) / "Tiles074_NormalDX.jpg",
        "roughness": Path(gt_extract_dir) / "Tiles074_Roughness.jpg"
    }

    for file_name in data.path_map:
        assert Path(data.path_map[file_name]) == gt_path_map[file_name]
        assert Path(data.path_map[file_name]).is_file()

    assert Path(data.albedo_texture_path) == gt_path_map["albedo"]
    assert Path(data.normal_texture_path) == gt_path_map["normal"]
    assert Path(data.roughness_texture_path) == gt_path_map["roughness"]

    assert data.prefix == gt_prefix
    assert Path(data.data_root) == gt_data_root
    assert Path(data.download_dir) == gt_download_dir
    assert Path(data.extract_dir) == gt_extract_dir


def test_terrazzo_texture():
    gt_prefix = "TerrazzoTexture"
    gt_data_root, gt_download_dir, gt_extract_dir = get_test_data_dirs(
        gt_prefix)

    data = o3d.data.TerrazzoTexture()
    assert Path(gt_download_dir).is_dir()

    gt_path_map = {
        "albedo": Path(gt_extract_dir) / "Terrazzo018_Color.jpg",
        "normal": Path(gt_extract_dir) / "Terrazzo018_NormalDX.jpg",
        "roughness": Path(gt_extract_dir) / "Terrazzo018_Roughness.jpg"
    }

    for file_name in data.path_map:
        assert Path(data.path_map[file_name]) == gt_path_map[file_name]
        assert Path(data.path_map[file_name]).is_file()

    assert Path(data.albedo_texture_path) == gt_path_map["albedo"]
    assert Path(data.normal_texture_path) == gt_path_map["normal"]
    assert Path(data.roughness_texture_path) == gt_path_map["roughness"]

    assert data.prefix == gt_prefix
    assert Path(data.data_root) == gt_data_root
    assert Path(data.download_dir) == gt_download_dir
    assert Path(data.extract_dir) == gt_extract_dir


def test_wood_texture():
    gt_prefix = "WoodTexture"
    gt_data_root, gt_download_dir, gt_extract_dir = get_test_data_dirs(
        gt_prefix)

    data = o3d.data.WoodTexture()
    assert Path(gt_download_dir).is_dir()

    gt_path_map = {
        "albedo": Path(gt_extract_dir) / "Wood049_Color.jpg",
        "normal": Path(gt_extract_dir) / "Wood049_NormalDX.jpg",
        "roughness": Path(gt_extract_dir) / "Wood049_Roughness.jpg"
    }

    for file_name in data.path_map:
        assert Path(data.path_map[file_name]) == gt_path_map[file_name]
        assert Path(data.path_map[file_name]).is_file()

    assert Path(data.albedo_texture_path) == gt_path_map["albedo"]
    assert Path(data.normal_texture_path) == gt_path_map["normal"]
    assert Path(data.roughness_texture_path) == gt_path_map["roughness"]

    assert data.prefix == gt_prefix
    assert Path(data.data_root) == gt_data_root
    assert Path(data.download_dir) == gt_download_dir
    assert Path(data.extract_dir) == gt_extract_dir


def test_wood_floor_texture():
    gt_prefix = "WoodFloorTexture"
    gt_data_root, gt_download_dir, gt_extract_dir = get_test_data_dirs(
        gt_prefix)

    data = o3d.data.WoodFloorTexture()
    assert Path(gt_download_dir).is_dir()

    gt_path_map = {
        "albedo": Path(gt_extract_dir) / "WoodFloor050_Color.jpg",
        "normal": Path(gt_extract_dir) / "WoodFloor050_NormalDX.jpg",
        "roughness": Path(gt_extract_dir) / "WoodFloor050_Roughness.jpg"
    }

    for file_name in data.path_map:
        assert Path(data.path_map[file_name]) == gt_path_map[file_name]
        assert Path(data.path_map[file_name]).is_file()

    assert Path(data.albedo_texture_path) == gt_path_map["albedo"]
    assert Path(data.normal_texture_path) == gt_path_map["normal"]
    assert Path(data.roughness_texture_path) == gt_path_map["roughness"]

    assert data.prefix == gt_prefix
    assert Path(data.data_root) == gt_data_root
    assert Path(data.download_dir) == gt_download_dir
    assert Path(data.extract_dir) == gt_extract_dir


def test_juneau():
    gt_prefix = "JuneauImage"
    gt_data_root, gt_download_dir, gt_extract_dir = get_test_data_dirs(
        gt_prefix)

    juneau = o3d.data.JuneauImage()
    assert Path(gt_download_dir).is_dir()

    assert Path(juneau.path) == gt_extract_dir / "JuneauImage.jpg"
    assert Path(juneau.path).is_file()

    assert juneau.prefix == gt_prefix
    assert Path(juneau.data_root) == gt_data_root
    assert Path(juneau.download_dir) == gt_download_dir
    assert Path(juneau.extract_dir) == gt_extract_dir
