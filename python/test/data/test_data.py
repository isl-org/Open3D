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
import pytest

from pathlib import Path
import os
import shutil


def test_dataset_base():
    default_data_root = Path(os.path.join(Path.home(), "open3d_data"))

    ds = o3d.data.Dataset("some_prefix")
    assert Path(ds.data_root) == default_data_root

    ds_custom = o3d.data.Dataset("some_prefix", "/my/custom/data_root")
    assert ds_custom.data_root == "/my/custom/data_root"
    assert ds_custom.prefix == "some_prefix"
    assert Path(ds_custom.download_dir) == Path(
        "/my/custom/data_root/download/some_prefix")
    assert Path(ds_custom.extract_dir) == Path(
        "/my/custom/data_root/extract/some_prefix")


def test_simple_dataset_base():
    prefix = "O3DTestSimpleDataset"
    data_root = Path(os.path.join(Path.home(), "open3d_data"))
    download_dir = Path(os.path.join(data_root, "download", prefix))
    extract_dir = Path(os.path.join(data_root, "extract", prefix))
    url_mirrors = [
        "https://github.com/isl-org/open3d_downloads/releases/download/"
        "20220201-data/BunnyMesh.ply"
    ]
    md5 = "568f871d1a221ba6627569f1e6f9a3f2"

    shutil.rmtree(download_dir, ignore_errors=True)
    shutil.rmtree(extract_dir, ignore_errors=True)

    single_download_dataset = o3d.data.SingleDownloadDataset(prefix,
                                                             url_mirrors,
                                                             md5,
                                                             no_extract=True)

    assert os.path.isfile(os.path.join(extract_dir, "BunnyMesh.ply")) == True
    assert os.path.isfile(os.path.join(download_dir, "BunnyMesh.ply")) == True

    assert single_download_dataset.prefix == prefix
    assert Path(single_download_dataset.data_root) == Path(data_root)
    assert Path(single_download_dataset.download_dir) == Path(download_dir)
    assert Path(single_download_dataset.extract_dir) == Path(extract_dir)

    shutil.rmtree(download_dir, ignore_errors=True)
    shutil.rmtree(extract_dir, ignore_errors=True)


def test_demo_icp_pointclouds():
    prefix = "O3DTestDemoICPPointClouds"
    data_root = Path(os.path.join(Path.home(), "open3d_data"))
    download_dir = Path(os.path.join(data_root, "download", prefix))
    extract_dir = Path(os.path.join(data_root, "extract", prefix))

    shutil.rmtree(download_dir, ignore_errors=True)
    shutil.rmtree(extract_dir, ignore_errors=True)

    demo_icp = o3d.data.DemoICPPointClouds(prefix)
    assert os.path.isdir(download_dir) == True

    paths = [
        os.path.join(extract_dir, "cloud_bin_0.pcd"),
        os.path.join(extract_dir, "cloud_bin_1.pcd"),
        os.path.join(extract_dir, "cloud_bin_2.pcd")
    ]

    assert demo_icp.paths == paths
    for path in demo_icp.paths:
        assert os.path.isfile(path) == True

    assert demo_icp.prefix == prefix
    assert Path(demo_icp.data_root) == Path(data_root)
    assert Path(demo_icp.download_dir) == Path(download_dir)
    assert Path(demo_icp.extract_dir) == Path(extract_dir)

    shutil.rmtree(download_dir, ignore_errors=True)
    shutil.rmtree(extract_dir, ignore_errors=True)


def test_demo_colored_icp_pointclouds():
    prefix = "O3DTestDemoColoredICPPointClouds"
    data_root = Path(os.path.join(Path.home(), "open3d_data"))
    download_dir = Path(os.path.join(data_root, "download", prefix))
    extract_dir = Path(os.path.join(data_root, "extract", prefix))

    shutil.rmtree(download_dir, ignore_errors=True)
    shutil.rmtree(extract_dir, ignore_errors=True)

    demo_colored_icp = o3d.data.DemoColoredICPPointClouds(prefix)
    assert os.path.isdir(download_dir) == True

    paths = [
        os.path.join(extract_dir, "frag_115.ply"),
        os.path.join(extract_dir, "frag_116.ply")
    ]
    assert demo_colored_icp.paths == paths
    for path in demo_colored_icp.paths:
        assert os.path.isfile(path) == True

    assert demo_colored_icp.prefix == prefix
    assert Path(demo_colored_icp.data_root) == Path(data_root)
    assert Path(demo_colored_icp.download_dir) == Path(download_dir)
    assert Path(demo_colored_icp.extract_dir) == Path(extract_dir)

    shutil.rmtree(download_dir, ignore_errors=True)
    shutil.rmtree(extract_dir, ignore_errors=True)


def test_demo_crop_pointcloud():
    prefix = "O3DTestDemoCropPointCloud"
    data_root = Path(os.path.join(Path.home(), "open3d_data"))
    download_dir = Path(os.path.join(data_root, "download", prefix))
    extract_dir = Path(os.path.join(data_root, "extract", prefix))

    shutil.rmtree(download_dir, ignore_errors=True)
    shutil.rmtree(extract_dir, ignore_errors=True)

    demo_crop_pcd = o3d.data.DemoCropPointCloud(prefix)
    assert os.path.isdir(download_dir) == True

    assert Path(demo_crop_pcd.pointcloud_path) == Path(
        os.path.join(extract_dir, "fragment.ply"))
    assert os.path.isfile(demo_crop_pcd.pointcloud_path) == True
    assert Path(demo_crop_pcd.cropped_json_path) == Path(
        os.path.join(extract_dir, "cropped.json"))
    assert os.path.isfile(demo_crop_pcd.pointcloud_path) == True

    assert demo_crop_pcd.prefix == prefix
    assert Path(demo_crop_pcd.data_root) == Path(data_root)
    assert Path(demo_crop_pcd.download_dir) == Path(download_dir)
    assert Path(demo_crop_pcd.extract_dir) == Path(extract_dir)

    shutil.rmtree(download_dir, ignore_errors=True)
    shutil.rmtree(extract_dir, ignore_errors=True)


def test_demo_pointcloud_feature_matching():
    prefix = "O3DTestDemoPointCloudFeatureMatching"
    data_root = Path(os.path.join(Path.home(), "open3d_data"))
    download_dir = Path(os.path.join(data_root, "download", prefix))
    extract_dir = Path(os.path.join(data_root, "extract", prefix))

    shutil.rmtree(download_dir, ignore_errors=True)
    shutil.rmtree(extract_dir, ignore_errors=True)

    demo_feature_matching = o3d.data.DemoPointCloudFeatureMatching(prefix)
    assert os.path.isdir(download_dir) == True

    pointcloud_paths = [
        os.path.join(extract_dir, "cloud_bin_0.pcd"),
        os.path.join(extract_dir, "cloud_bin_1.pcd")
    ]
    assert demo_feature_matching.pointcloud_paths == pointcloud_paths
    assert os.path.isfile(demo_feature_matching.pointcloud_paths[0]) == True
    assert os.path.isfile(demo_feature_matching.pointcloud_paths[1]) == True

    fpfh_feature_paths = [
        os.path.join(extract_dir, "cloud_bin_0.fpfh.bin"),
        os.path.join(extract_dir, "cloud_bin_1.fpfh.bin")
    ]
    assert demo_feature_matching.fpfh_feature_paths == fpfh_feature_paths
    assert os.path.isfile(demo_feature_matching.fpfh_feature_paths[0]) == True
    assert os.path.isfile(demo_feature_matching.fpfh_feature_paths[1]) == True

    l32d_feature_paths = [
        os.path.join(extract_dir, "cloud_bin_0.d32.bin"),
        os.path.join(extract_dir, "cloud_bin_1.d32.bin")
    ]
    assert demo_feature_matching.l32d_feature_paths == l32d_feature_paths
    assert os.path.isfile(demo_feature_matching.l32d_feature_paths[0]) == True
    assert os.path.isfile(demo_feature_matching.l32d_feature_paths[1]) == True

    assert demo_feature_matching.prefix == prefix
    assert Path(demo_feature_matching.data_root) == Path(data_root)
    assert Path(demo_feature_matching.download_dir) == Path(download_dir)
    assert Path(demo_feature_matching.extract_dir) == Path(extract_dir)

    shutil.rmtree(download_dir, ignore_errors=True)
    shutil.rmtree(extract_dir, ignore_errors=True)


def test_demo_pose_graph_optimization():
    prefix = "O3DTestDemoPoseGraphOptimization"
    data_root = Path(os.path.join(Path.home(), "open3d_data"))
    download_dir = Path(os.path.join(data_root, "download", prefix))
    extract_dir = Path(os.path.join(data_root, "extract", prefix))

    shutil.rmtree(download_dir, ignore_errors=True)
    shutil.rmtree(extract_dir, ignore_errors=True)

    demo_pose_optimization = o3d.data.DemoPoseGraphOptimization(prefix)
    assert os.path.isdir(download_dir) == True

    assert demo_pose_optimization.pose_graph_fragment_path == os.path.join(
        extract_dir, "pose_graph_example_fragment.json")
    assert os.path.isfile(
        demo_pose_optimization.pose_graph_fragment_path) == True
    assert demo_pose_optimization.pose_graph_global_path == os.path.join(
        extract_dir, "pose_graph_example_global.json")
    assert os.path.isfile(demo_pose_optimization.pose_graph_global_path) == True

    assert demo_pose_optimization.prefix == prefix
    assert Path(demo_pose_optimization.data_root) == Path(data_root)
    assert Path(demo_pose_optimization.download_dir) == Path(download_dir)
    assert Path(demo_pose_optimization.extract_dir) == Path(extract_dir)

    shutil.rmtree(download_dir, ignore_errors=True)
    shutil.rmtree(extract_dir, ignore_errors=True)


def test_sample_pointcloud_pcd():
    prefix = "O3DTestSamplePointCloudPCD"
    data_root = Path(os.path.join(Path.home(), "open3d_data"))
    download_dir = Path(os.path.join(data_root, "download", prefix))
    extract_dir = Path(os.path.join(data_root, "extract", prefix))

    shutil.rmtree(download_dir, ignore_errors=True)
    shutil.rmtree(extract_dir, ignore_errors=True)

    pcd_pointcloud = o3d.data.SamplePointCloudPCD(prefix)
    assert os.path.isdir(download_dir) == True

    assert pcd_pointcloud.path == os.path.join(extract_dir, "fragment.pcd")
    assert os.path.isfile(pcd_pointcloud.path) == True

    assert pcd_pointcloud.prefix == prefix
    assert Path(pcd_pointcloud.data_root) == Path(data_root)
    assert Path(pcd_pointcloud.download_dir) == Path(download_dir)
    assert Path(pcd_pointcloud.extract_dir) == Path(extract_dir)

    shutil.rmtree(download_dir, ignore_errors=True)
    shutil.rmtree(extract_dir, ignore_errors=True)


def test_sample_pointcloud_ply():
    prefix = "O3DTestSamplePointCloudPLY"
    data_root = Path(os.path.join(Path.home(), "open3d_data"))
    download_dir = Path(os.path.join(data_root, "download", prefix))
    extract_dir = Path(os.path.join(data_root, "extract", prefix))

    shutil.rmtree(download_dir, ignore_errors=True)
    shutil.rmtree(extract_dir, ignore_errors=True)

    ply_pointcloud = o3d.data.SamplePointCloudPLY(prefix)
    assert os.path.isdir(download_dir) == True

    assert ply_pointcloud.path == os.path.join(extract_dir, "fragment.ply")
    assert os.path.isfile(ply_pointcloud.path) == True

    assert ply_pointcloud.prefix == prefix
    assert Path(ply_pointcloud.data_root) == Path(data_root)
    assert Path(ply_pointcloud.download_dir) == Path(download_dir)
    assert Path(ply_pointcloud.extract_dir) == Path(extract_dir)

    shutil.rmtree(download_dir, ignore_errors=True)
    shutil.rmtree(extract_dir, ignore_errors=True)


def test_sample_rgbd_image_nyu():
    prefix = "O3DTestSampleRGBDImageNYU"
    data_root = Path(os.path.join(Path.home(), "open3d_data"))
    download_dir = Path(os.path.join(data_root, "download", prefix))
    extract_dir = Path(os.path.join(data_root, "extract", prefix))

    shutil.rmtree(download_dir, ignore_errors=True)
    shutil.rmtree(extract_dir, ignore_errors=True)

    rgbd_image_nyu = o3d.data.SampleRGBDImageNYU(prefix)
    assert os.path.isdir(download_dir) == True

    assert rgbd_image_nyu.color_path == os.path.join(extract_dir,
                                                     "NYU_color.ppm")
    assert os.path.isfile(rgbd_image_nyu.color_path) == True

    assert rgbd_image_nyu.depth_path == os.path.join(extract_dir,
                                                     "NYU_depth.pgm")
    assert os.path.isfile(rgbd_image_nyu.depth_path) == True

    assert rgbd_image_nyu.prefix == prefix
    assert Path(rgbd_image_nyu.data_root) == Path(data_root)
    assert Path(rgbd_image_nyu.download_dir) == Path(download_dir)
    assert Path(rgbd_image_nyu.extract_dir) == Path(extract_dir)

    shutil.rmtree(download_dir, ignore_errors=True)
    shutil.rmtree(extract_dir, ignore_errors=True)


def test_sample_rgbd_image_sun():
    prefix = "O3DTestSampleRGBDImageSUN"
    data_root = Path(os.path.join(Path.home(), "open3d_data"))
    download_dir = Path(os.path.join(data_root, "download", prefix))
    extract_dir = Path(os.path.join(data_root, "extract", prefix))

    shutil.rmtree(download_dir, ignore_errors=True)
    shutil.rmtree(extract_dir, ignore_errors=True)

    rgbd_image_sun = o3d.data.SampleRGBDImageSUN(prefix)
    assert os.path.isdir(download_dir) == True

    assert rgbd_image_sun.color_path == os.path.join(extract_dir,
                                                     "SUN_color.jpg")
    assert os.path.isfile(rgbd_image_sun.color_path) == True

    assert rgbd_image_sun.depth_path == os.path.join(extract_dir,
                                                     "SUN_depth.png")
    assert os.path.isfile(rgbd_image_sun.depth_path) == True

    assert rgbd_image_sun.prefix == prefix
    assert Path(rgbd_image_sun.data_root) == Path(data_root)
    assert Path(rgbd_image_sun.download_dir) == Path(download_dir)
    assert Path(rgbd_image_sun.extract_dir) == Path(extract_dir)

    shutil.rmtree(download_dir, ignore_errors=True)
    shutil.rmtree(extract_dir, ignore_errors=True)


def test_sample_rgbd_image_tum():
    prefix = "O3DTestSampleRGBDImageTUM"
    data_root = Path(os.path.join(Path.home(), "open3d_data"))
    download_dir = Path(os.path.join(data_root, "download", prefix))
    extract_dir = Path(os.path.join(data_root, "extract", prefix))

    shutil.rmtree(download_dir, ignore_errors=True)
    shutil.rmtree(extract_dir, ignore_errors=True)

    rgbd_image_tum = o3d.data.SampleRGBDImageTUM(prefix)
    assert os.path.isdir(download_dir) == True

    assert rgbd_image_tum.color_path == os.path.join(extract_dir,
                                                     "TUM_color.png")
    assert os.path.isfile(rgbd_image_tum.color_path) == True

    assert rgbd_image_tum.depth_path == os.path.join(extract_dir,
                                                     "TUM_depth.png")
    assert os.path.isfile(rgbd_image_tum.depth_path) == True

    assert rgbd_image_tum.prefix == prefix
    assert Path(rgbd_image_tum.data_root) == Path(data_root)
    assert Path(rgbd_image_tum.download_dir) == Path(download_dir)
    assert Path(rgbd_image_tum.extract_dir) == Path(extract_dir)

    shutil.rmtree(download_dir, ignore_errors=True)
    shutil.rmtree(extract_dir, ignore_errors=True)


def test_sample_rgbd_dataset_icl():
    prefix = "O3DTestSampleRGBDDatasetICL"
    data_root = Path(os.path.join(Path.home(), "open3d_data"))
    download_dir = Path(os.path.join(data_root, "download", prefix))
    extract_dir = Path(os.path.join(data_root, "extract", prefix))

    shutil.rmtree(download_dir, ignore_errors=True)
    shutil.rmtree(extract_dir, ignore_errors=True)

    rgbd_dataset_icl = o3d.data.SampleRGBDDatasetICL(prefix)
    assert os.path.isdir(download_dir) == True

    color_paths = [
        os.path.join(extract_dir, "color", "00000.jpg"),
        os.path.join(extract_dir, "color", "00001.jpg"),
        os.path.join(extract_dir, "color", "00002.jpg"),
        os.path.join(extract_dir, "color", "00003.jpg"),
        os.path.join(extract_dir, "color", "00004.jpg")
    ]

    assert rgbd_dataset_icl.color_paths == color_paths
    for path in rgbd_dataset_icl.color_paths:
        assert os.path.isfile(path) == True

    depth_paths = [
        os.path.join(extract_dir, "depth", "00000.png"),
        os.path.join(extract_dir, "depth", "00001.png"),
        os.path.join(extract_dir, "depth", "00002.png"),
        os.path.join(extract_dir, "depth", "00003.png"),
        os.path.join(extract_dir, "depth", "00004.png")
    ]
    assert rgbd_dataset_icl.depth_paths == depth_paths
    for path in rgbd_dataset_icl.depth_paths:
        assert os.path.isfile(path) == True

    assert Path(rgbd_dataset_icl.trajectory_log_path) == Path(
        os.path.join(extract_dir, "trajectory.log"))
    assert Path(rgbd_dataset_icl.odometry_log_path) == Path(
        os.path.join(extract_dir, "odometry.log"))
    assert Path(rgbd_dataset_icl.rgbd_match_path) == Path(
        os.path.join(extract_dir, "rgbd.match"))
    assert Path(rgbd_dataset_icl.reconstruction_path) == Path(
        os.path.join(extract_dir, "example_tsdf_pcd.ply"))

    assert rgbd_dataset_icl.prefix == prefix
    assert Path(rgbd_dataset_icl.data_root) == Path(data_root)
    assert Path(rgbd_dataset_icl.download_dir) == Path(download_dir)
    assert Path(rgbd_dataset_icl.extract_dir) == Path(extract_dir)

    shutil.rmtree(download_dir, ignore_errors=True)
    shutil.rmtree(extract_dir, ignore_errors=True)


def test_sample_fountain_rgbd_dataset():
    prefix = "O3DTestSampleFountainRGBDDataset"
    data_root = Path(os.path.join(Path.home(), "open3d_data"))
    download_dir = Path(os.path.join(data_root, "download", prefix))
    extract_dir = Path(os.path.join(data_root, "extract", prefix))

    shutil.rmtree(download_dir, ignore_errors=True)
    shutil.rmtree(extract_dir, ignore_errors=True)

    fountain_dataset = o3d.data.SampleFountainRGBDDataset(prefix)
    assert os.path.isdir(download_dir) == True

    color_paths = [
        os.path.join(extract_dir, "image", "0000010-000001228920.jpg"),
        os.path.join(extract_dir, "image", "0000031-000004096400.jpg"),
        os.path.join(extract_dir, "image", "0000044-000005871507.jpg"),
        os.path.join(extract_dir, "image", "0000064-000008602440.jpg"),
        os.path.join(extract_dir, "image", "0000110-000014883587.jpg"),
        os.path.join(extract_dir, "image", "0000156-000021164733.jpg"),
        os.path.join(extract_dir, "image", "0000200-000027172787.jpg"),
        os.path.join(extract_dir, "image", "0000215-000029220987.jpg"),
        os.path.join(extract_dir, "image", "0000255-000034682853.jpg"),
        os.path.join(extract_dir, "image", "0000299-000040690907.jpg"),
        os.path.join(extract_dir, "image", "0000331-000045060400.jpg"),
        os.path.join(extract_dir, "image", "0000368-000050112627.jpg"),
        os.path.join(extract_dir, "image", "0000412-000056120680.jpg"),
        os.path.join(extract_dir, "image", "0000429-000058441973.jpg"),
        os.path.join(extract_dir, "image", "0000474-000064586573.jpg"),
        os.path.join(extract_dir, "image", "0000487-000066361680.jpg"),
        os.path.join(extract_dir, "image", "0000526-000071687000.jpg"),
        os.path.join(extract_dir, "image", "0000549-000074827573.jpg"),
        os.path.join(extract_dir, "image", "0000582-000079333613.jpg"),
        os.path.join(extract_dir, "image", "0000630-000085887853.jpg"),
        os.path.join(extract_dir, "image", "0000655-000089301520.jpg"),
        os.path.join(extract_dir, "image", "0000703-000095855760.jpg"),
        os.path.join(extract_dir, "image", "0000722-000098450147.jpg"),
        os.path.join(extract_dir, "image", "0000771-000105140933.jpg"),
        os.path.join(extract_dir, "image", "0000792-000108008413.jpg"),
        os.path.join(extract_dir, "image", "0000818-000111558627.jpg"),
        os.path.join(extract_dir, "image", "0000849-000115791573.jpg"),
        os.path.join(extract_dir, "image", "0000883-000120434160.jpg"),
        os.path.join(extract_dir, "image", "0000896-000122209267.jpg"),
        os.path.join(extract_dir, "image", "0000935-000127534587.jpg"),
        os.path.join(extract_dir, "image", "0000985-000134361920.jpg"),
        os.path.join(extract_dir, "image", "0001028-000140233427.jpg"),
        os.path.join(extract_dir, "image", "0001061-000144739467.jpg")
    ]
    assert fountain_dataset.color_paths == color_paths
    for path in fountain_dataset.color_paths:
        assert os.path.isfile(path) == True

    depth_paths = [
        os.path.join(extract_dir, "depth", "0000038-000001234662.png"),
        os.path.join(extract_dir, "depth", "0000124-000004104418.png"),
        os.path.join(extract_dir, "depth", "0000177-000005872988.png"),
        os.path.join(extract_dir, "depth", "0000259-000008609267.png"),
        os.path.join(extract_dir, "depth", "0000447-000014882686.png"),
        os.path.join(extract_dir, "depth", "0000635-000021156105.png"),
        os.path.join(extract_dir, "depth", "0000815-000027162570.png"),
        os.path.join(extract_dir, "depth", "0000877-000029231463.png"),
        os.path.join(extract_dir, "depth", "0001040-000034670651.png"),
        os.path.join(extract_dir, "depth", "0001220-000040677116.png"),
        os.path.join(extract_dir, "depth", "0001351-000045048488.png"),
        os.path.join(extract_dir, "depth", "0001503-000050120614.png"),
        os.path.join(extract_dir, "depth", "0001683-000056127079.png"),
        os.path.join(extract_dir, "depth", "0001752-000058429557.png"),
        os.path.join(extract_dir, "depth", "0001937-000064602868.png"),
        os.path.join(extract_dir, "depth", "0001990-000066371438.png"),
        os.path.join(extract_dir, "depth", "0002149-000071677149.png"),
        os.path.join(extract_dir, "depth", "0002243-000074813859.png"),
        os.path.join(extract_dir, "depth", "0002378-000079318707.png"),
        os.path.join(extract_dir, "depth", "0002575-000085892450.png"),
        os.path.join(extract_dir, "depth", "0002677-000089296113.png"),
        os.path.join(extract_dir, "depth", "0002874-000095869855.png"),
        os.path.join(extract_dir, "depth", "0002951-000098439288.png"),
        os.path.join(extract_dir, "depth", "0003152-000105146507.png"),
        os.path.join(extract_dir, "depth", "0003238-000108016262.png"),
        os.path.join(extract_dir, "depth", "0003344-000111553403.png"),
        os.path.join(extract_dir, "depth", "0003471-000115791298.png"),
        os.path.join(extract_dir, "depth", "0003610-000120429623.png"),
        os.path.join(extract_dir, "depth", "0003663-000122198194.png"),
        os.path.join(extract_dir, "depth", "0003823-000127537274.png"),
        os.path.join(extract_dir, "depth", "0004028-000134377970.png"),
        os.path.join(extract_dir, "depth", "0004203-000140217589.png"),
        os.path.join(extract_dir, "depth", "0004339-000144755807.png")
    ]
    assert fountain_dataset.depth_paths == depth_paths
    for path in fountain_dataset.depth_paths:
        assert os.path.isfile(path) == True

    assert Path(fountain_dataset.keyframe_poses_log_path) == Path(
        os.path.join(extract_dir, "scene", "key.log"))
    assert Path(fountain_dataset.reconstruction_path) == Path(
        os.path.join(extract_dir, "scene", "integrated.ply"))

    assert fountain_dataset.prefix == prefix
    assert Path(fountain_dataset.data_root) == Path(data_root)
    assert Path(fountain_dataset.download_dir) == Path(download_dir)
    assert Path(fountain_dataset.extract_dir) == Path(extract_dir)

    shutil.rmtree(download_dir, ignore_errors=True)
    shutil.rmtree(extract_dir, ignore_errors=True)


def test_eagle():
    prefix = "O3DTestEagle"
    data_root = Path(os.path.join(Path.home(), "open3d_data"))
    download_dir = Path(os.path.join(data_root, "download", prefix))
    extract_dir = Path(os.path.join(data_root, "extract", prefix))

    shutil.rmtree(download_dir, ignore_errors=True)
    shutil.rmtree(extract_dir, ignore_errors=True)

    eagle = o3d.data.EaglePointCloud(prefix)
    assert os.path.isdir(download_dir) == True

    assert Path(eagle.path) == Path(
        os.path.join(extract_dir, "EaglePointCloud.ply"))
    assert os.path.isfile(eagle.path) == True

    assert eagle.prefix == prefix
    assert Path(eagle.data_root) == Path(data_root)
    assert Path(eagle.download_dir) == Path(download_dir)
    assert Path(eagle.extract_dir) == Path(extract_dir)

    shutil.rmtree(download_dir, ignore_errors=True)
    shutil.rmtree(extract_dir, ignore_errors=True)


def test_armadillo():
    prefix = "O3DTestArmadillo"
    data_root = Path(os.path.join(Path.home(), "open3d_data"))
    download_dir = Path(os.path.join(data_root, "download", prefix))
    extract_dir = Path(os.path.join(data_root, "extract", prefix))

    shutil.rmtree(download_dir, ignore_errors=True)
    shutil.rmtree(extract_dir, ignore_errors=True)

    armadillo = o3d.data.ArmadilloMesh(prefix)
    assert os.path.isdir(download_dir) == True

    assert Path(armadillo.path) == Path(
        os.path.join(extract_dir, "ArmadilloMesh.ply"))
    assert os.path.isfile(armadillo.path) == True

    assert armadillo.prefix == prefix
    assert Path(armadillo.data_root) == Path(data_root)
    assert Path(armadillo.download_dir) == Path(download_dir)
    assert Path(armadillo.extract_dir) == Path(extract_dir)

    shutil.rmtree(download_dir, ignore_errors=True)
    shutil.rmtree(extract_dir, ignore_errors=True)


def test_bunny():
    prefix = "O3DTestBunny"
    data_root = Path(os.path.join(Path.home(), "open3d_data"))
    download_dir = Path(os.path.join(data_root, "download", prefix))
    extract_dir = Path(os.path.join(data_root, "extract", prefix))

    shutil.rmtree(download_dir, ignore_errors=True)
    shutil.rmtree(extract_dir, ignore_errors=True)

    bunny = o3d.data.BunnyMesh(prefix)
    assert os.path.isdir(download_dir) == True

    assert Path(bunny.path) == Path(os.path.join(extract_dir, "BunnyMesh.ply"))
    assert os.path.isfile(bunny.path) == True

    assert bunny.prefix == prefix
    assert Path(bunny.data_root) == Path(data_root)
    assert Path(bunny.download_dir) == Path(download_dir)
    assert Path(bunny.extract_dir) == Path(extract_dir)

    shutil.rmtree(download_dir, ignore_errors=True)
    shutil.rmtree(extract_dir, ignore_errors=True)


def test_knot():
    prefix = "O3DTestKnot"
    data_root = Path(os.path.join(Path.home(), "open3d_data"))
    download_dir = Path(os.path.join(data_root, "download", prefix))
    extract_dir = Path(os.path.join(data_root, "extract", prefix))

    shutil.rmtree(download_dir, ignore_errors=True)
    shutil.rmtree(extract_dir, ignore_errors=True)

    knot = o3d.data.KnotMesh(prefix)
    assert os.path.isdir(download_dir) == True

    assert Path(knot.path) == Path(os.path.join(extract_dir, "KnotMesh.ply"))
    assert os.path.isfile(knot.path) == True

    assert knot.prefix == prefix
    assert Path(knot.data_root) == Path(data_root)
    assert Path(knot.download_dir) == Path(download_dir)
    assert Path(knot.extract_dir) == Path(extract_dir)

    shutil.rmtree(download_dir, ignore_errors=True)
    shutil.rmtree(extract_dir, ignore_errors=True)


def test_juneau():
    prefix = "O3DTestJuneau"
    data_root = Path(os.path.join(Path.home(), "open3d_data"))
    download_dir = Path(os.path.join(data_root, "download", prefix))
    extract_dir = Path(os.path.join(data_root, "extract", prefix))

    shutil.rmtree(download_dir, ignore_errors=True)
    shutil.rmtree(extract_dir, ignore_errors=True)

    juneau = o3d.data.JuneauImage(prefix)
    assert os.path.isdir(download_dir) == True

    assert Path(juneau.path) == Path(
        os.path.join(extract_dir, "JuneauImage.jpg"))
    assert os.path.isfile(juneau.path) == True

    assert juneau.prefix == prefix
    assert Path(juneau.data_root) == Path(data_root)
    assert Path(juneau.download_dir) == Path(download_dir)
    assert Path(juneau.extract_dir) == Path(extract_dir)

    shutil.rmtree(download_dir, ignore_errors=True)
    shutil.rmtree(extract_dir, ignore_errors=True)
