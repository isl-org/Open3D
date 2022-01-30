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
    default_data_root = os.path.join(Path.home(), "open3d_data")

    ds = o3d.data.Dataset("some_prefix")
    assert ds.data_root == default_data_root

    ds_custom = o3d.data.Dataset("some_prefix", "/my/custom/data_root")
    assert ds_custom.data_root == "/my/custom/data_root"
    assert ds_custom.prefix == "some_prefix"
    assert ds_custom.download_dir == "/my/custom/data_root/download/some_prefix"
    assert ds_custom.extract_dir == "/my/custom/data_root/extract/some_prefix"


def test_simple_dataset_base():
    prefix = "O3DTestSimpleDataset"
    data_root = os.path.join(Path.home(), "open3d_data")
    download_dir = os.path.join(data_root, "download", prefix)
    extract_dir = os.path.join(data_root, "extract", prefix)
    url_mirrors = [
        "https://github.com/isl-org/open3d_downloads/releases/download/"
        "290122-sample-meshs/BunnyMesh.ply"
    ]
    md5 = "568f871d1a221ba6627569f1e6f9a3f2"

    shutil.rmtree(download_dir, ignore_errors=True)
    shutil.rmtree(extract_dir, ignore_errors=True)

    simple_dataset = o3d.data.SimpleDataset(prefix,
                                            url_mirrors,
                                            md5,
                                            no_extract=True)

    assert os.path.isfile(extract_dir + "/BunnyMesh.ply") == True
    assert os.path.isfile(download_dir + "/BunnyMesh.ply") == True

    assert simple_dataset.prefix == prefix
    assert simple_dataset.data_root == data_root
    assert simple_dataset.download_dir == download_dir
    assert simple_dataset.extract_dir == extract_dir

    shutil.rmtree(download_dir, ignore_errors=True)
    shutil.rmtree(extract_dir, ignore_errors=True)


def test_demo_icp_pointclouds():
    prefix = "O3DTestDemoICPPointClouds"
    data_root = os.path.join(Path.home(), "open3d_data")
    download_dir = os.path.join(data_root, "download", prefix)
    extract_dir = os.path.join(data_root, "extract", prefix)

    shutil.rmtree(download_dir, ignore_errors=True)
    shutil.rmtree(extract_dir, ignore_errors=True)

    demo_icp = o3d.data.DemoICPPointClouds(prefix)
    assert os.path.isdir(download_dir) == True

    paths = [
        extract_dir + "/cloud_bin_0.pcd", extract_dir + "/cloud_bin_1.pcd",
        extract_dir + "/cloud_bin_2.pcd"
    ]
    assert demo_icp.paths == paths
    for path in demo_icp.paths:
        assert os.path.isfile(path) == True

    assert demo_icp.prefix == prefix
    assert demo_icp.data_root == data_root
    assert demo_icp.download_dir == download_dir
    assert demo_icp.extract_dir == extract_dir

    shutil.rmtree(download_dir, ignore_errors=True)
    shutil.rmtree(extract_dir, ignore_errors=True)


def test_demo_colored_icp_pointclouds():
    prefix = "O3DTestDemoColoredICPPointClouds"
    data_root = os.path.join(Path.home(), "open3d_data")
    download_dir = os.path.join(data_root, "download", prefix)
    extract_dir = os.path.join(data_root, "extract", prefix)

    shutil.rmtree(download_dir, ignore_errors=True)
    shutil.rmtree(extract_dir, ignore_errors=True)

    demo_colored_icp = o3d.data.DemoColoredICPPointClouds(prefix)
    assert os.path.isdir(download_dir) == True

    paths = [extract_dir + "/frag_115.ply", extract_dir + "/frag_116.ply"]
    assert demo_colored_icp.paths == paths
    for path in demo_colored_icp.paths:
        assert os.path.isfile(path) == True

    assert demo_colored_icp.prefix == prefix
    assert demo_colored_icp.data_root == data_root
    assert demo_colored_icp.download_dir == download_dir
    assert demo_colored_icp.extract_dir == extract_dir

    shutil.rmtree(download_dir, ignore_errors=True)
    shutil.rmtree(extract_dir, ignore_errors=True)


def test_demo_crop_pointcloud():
    prefix = "O3DTestDemoCropPointCloud"
    data_root = os.path.join(Path.home(), "open3d_data")
    download_dir = os.path.join(data_root, "download", prefix)
    extract_dir = os.path.join(data_root, "extract", prefix)

    shutil.rmtree(download_dir, ignore_errors=True)
    shutil.rmtree(extract_dir, ignore_errors=True)

    demo_crop_pcd = o3d.data.DemoCropPointCloud(prefix)
    assert os.path.isdir(download_dir) == True

    assert demo_crop_pcd.path_pointcloud == extract_dir + "/fragment.ply"
    assert os.path.isfile(demo_crop_pcd.path_pointcloud) == True
    assert demo_crop_pcd.path_cropped_json == extract_dir + "/cropped.json"
    assert os.path.isfile(demo_crop_pcd.path_pointcloud) == True

    assert demo_crop_pcd.prefix == prefix
    assert demo_crop_pcd.data_root == data_root
    assert demo_crop_pcd.download_dir == download_dir
    assert demo_crop_pcd.extract_dir == extract_dir

    shutil.rmtree(download_dir, ignore_errors=True)
    shutil.rmtree(extract_dir, ignore_errors=True)


def test_demo_pointcloud_feature_matching():
    prefix = "O3DTestDemoPointCloudFeatureMatching"
    data_root = os.path.join(Path.home(), "open3d_data")
    download_dir = os.path.join(data_root, "download", prefix)
    extract_dir = os.path.join(data_root, "extract", prefix)

    shutil.rmtree(download_dir, ignore_errors=True)
    shutil.rmtree(extract_dir, ignore_errors=True)

    demo_feature_matching = o3d.data.DemoPointCloudFeatureMatching(prefix)
    assert os.path.isdir(download_dir) == True

    paths_pointclouds = [
        extract_dir + "/cloud_bin_0.pcd", extract_dir + "/cloud_bin_1.pcd"
    ]
    assert demo_feature_matching.paths_pointclouds == paths_pointclouds
    assert os.path.isfile(demo_feature_matching.paths_pointclouds[0]) == True
    assert os.path.isfile(demo_feature_matching.paths_pointclouds[1]) == True

    paths_fpfh_features = [
        extract_dir + "/cloud_bin_0.fpfh.bin",
        extract_dir + "/cloud_bin_1.fpfh.bin"
    ]
    assert demo_feature_matching.paths_fpfh_features == paths_fpfh_features
    assert os.path.isfile(demo_feature_matching.paths_fpfh_features[0]) == True
    assert os.path.isfile(demo_feature_matching.paths_fpfh_features[1]) == True

    paths_l32d_features = [
        extract_dir + "/cloud_bin_0.d32.bin",
        extract_dir + "/cloud_bin_1.d32.bin"
    ]
    assert demo_feature_matching.paths_l32d_features == paths_l32d_features
    assert os.path.isfile(demo_feature_matching.paths_l32d_features[0]) == True
    assert os.path.isfile(demo_feature_matching.paths_l32d_features[1]) == True

    assert demo_feature_matching.prefix == prefix
    assert demo_feature_matching.data_root == data_root
    assert demo_feature_matching.download_dir == download_dir
    assert demo_feature_matching.extract_dir == extract_dir

    shutil.rmtree(download_dir, ignore_errors=True)
    shutil.rmtree(extract_dir, ignore_errors=True)


def test_demo_pose_graph_optimization():
    prefix = "O3DTestDemoPoseGraphOptimization"
    data_root = os.path.join(Path.home(), "open3d_data")
    download_dir = os.path.join(data_root, "download", prefix)
    extract_dir = os.path.join(data_root, "extract", prefix)

    shutil.rmtree(download_dir, ignore_errors=True)
    shutil.rmtree(extract_dir, ignore_errors=True)

    demo_pose_optimization = o3d.data.DemoPoseGraphOptimization(prefix)
    assert os.path.isdir(download_dir) == True

    assert demo_pose_optimization.path_pose_graph_fragment == extract_dir + \
        "/pose_graph_example_fragment.json"
    assert os.path.isfile(
        demo_pose_optimization.path_pose_graph_fragment) == True
    assert demo_pose_optimization.path_pose_graph_global == extract_dir + \
        "/pose_graph_example_global.json"
    assert os.path.isfile(demo_pose_optimization.path_pose_graph_global) == True

    assert demo_pose_optimization.prefix == prefix
    assert demo_pose_optimization.data_root == data_root
    assert demo_pose_optimization.download_dir == download_dir
    assert demo_pose_optimization.extract_dir == extract_dir

    shutil.rmtree(download_dir, ignore_errors=True)
    shutil.rmtree(extract_dir, ignore_errors=True)


def test_sample_pointcloud_pcd():
    prefix = "O3DTestSamplePointCloudPCD"
    data_root = os.path.join(Path.home(), "open3d_data")
    download_dir = os.path.join(data_root, "download", prefix)
    extract_dir = os.path.join(data_root, "extract", prefix)

    shutil.rmtree(download_dir, ignore_errors=True)
    shutil.rmtree(extract_dir, ignore_errors=True)

    pcd_pointcloud = o3d.data.SamplePointCloudPCD(prefix)
    assert os.path.isdir(download_dir) == True

    assert pcd_pointcloud.path == extract_dir + "/fragment.pcd"
    assert os.path.isfile(pcd_pointcloud.path) == True

    assert pcd_pointcloud.prefix == prefix
    assert pcd_pointcloud.data_root == data_root
    assert pcd_pointcloud.download_dir == download_dir
    assert pcd_pointcloud.extract_dir == extract_dir

    shutil.rmtree(download_dir, ignore_errors=True)
    shutil.rmtree(extract_dir, ignore_errors=True)


def test_sample_pointcloud_ply():
    prefix = "O3DTestSamplePointCloudPLY"
    data_root = os.path.join(Path.home(), "open3d_data")
    download_dir = os.path.join(data_root, "download", prefix)
    extract_dir = os.path.join(data_root, "extract", prefix)

    shutil.rmtree(download_dir, ignore_errors=True)
    shutil.rmtree(extract_dir, ignore_errors=True)

    ply_pointcloud = o3d.data.SamplePointCloudPCD(prefix)
    assert os.path.isdir(download_dir) == True

    assert ply_pointcloud.path == extract_dir + "/fragment.ply"
    assert os.path.isfile(ply_pointcloud.path) == True

    assert ply_pointcloud.prefix == prefix
    assert ply_pointcloud.data_root == data_root
    assert ply_pointcloud.download_dir == download_dir
    assert ply_pointcloud.extract_dir == extract_dir

    shutil.rmtree(download_dir, ignore_errors=True)
    shutil.rmtree(extract_dir, ignore_errors=True)


def test_eagle():
    prefix = "O3DTestEagle"
    data_root = os.path.join(Path.home(), "open3d_data")
    download_dir = os.path.join(data_root, "download", prefix)
    extract_dir = os.path.join(data_root, "extract", prefix)

    shutil.rmtree(download_dir, ignore_errors=True)
    shutil.rmtree(extract_dir, ignore_errors=True)

    eagle = o3d.data.Eagle(prefix)
    assert os.path.isdir(download_dir) == True

    assert eagle.path == extract_dir + "/EaglePointCloud.ply"
    assert os.path.isfile(eagle.path) == True

    assert eagle.prefix == prefix
    assert eagle.data_root == data_root
    assert eagle.download_dir == download_dir
    assert eagle.extract_dir == extract_dir

    shutil.rmtree(download_dir, ignore_errors=True)
    shutil.rmtree(extract_dir, ignore_errors=True)


def test_armadillo():
    prefix = "O3DTestArmadillo"
    data_root = os.path.join(Path.home(), "open3d_data")
    download_dir = os.path.join(data_root, "download", prefix)
    extract_dir = os.path.join(data_root, "extract", prefix)

    shutil.rmtree(download_dir, ignore_errors=True)
    shutil.rmtree(extract_dir, ignore_errors=True)

    armadillo = o3d.data.Armadillo(prefix)
    assert os.path.isdir(download_dir) == True

    assert armadillo.path == extract_dir + "/ArmadilloMesh.ply"
    assert os.path.isfile(armadillo.path) == True

    assert armadillo.prefix == prefix
    assert armadillo.data_root == data_root
    assert armadillo.download_dir == download_dir
    assert armadillo.extract_dir == extract_dir

    shutil.rmtree(download_dir, ignore_errors=True)
    shutil.rmtree(extract_dir, ignore_errors=True)


def test_bunny():
    prefix = "O3DTestBunny"
    data_root = os.path.join(Path.home(), "open3d_data")
    download_dir = os.path.join(data_root, "download", prefix)
    extract_dir = os.path.join(data_root, "extract", prefix)

    shutil.rmtree(download_dir, ignore_errors=True)
    shutil.rmtree(extract_dir, ignore_errors=True)

    bunny = o3d.data.Bunny(prefix)
    assert os.path.isdir(download_dir) == True

    assert bunny.path == extract_dir + "/BunnyMesh.ply"
    assert os.path.isfile(bunny.path) == True

    assert bunny.prefix == prefix
    assert bunny.data_root == data_root
    assert bunny.download_dir == download_dir
    assert bunny.extract_dir == extract_dir

    shutil.rmtree(download_dir, ignore_errors=True)
    shutil.rmtree(extract_dir, ignore_errors=True)


def test_knot():
    prefix = "O3DTestKnot"
    data_root = os.path.join(Path.home(), "open3d_data")
    download_dir = os.path.join(data_root, "download", prefix)
    extract_dir = os.path.join(data_root, "extract", prefix)

    shutil.rmtree(download_dir, ignore_errors=True)
    shutil.rmtree(extract_dir, ignore_errors=True)

    knot = o3d.data.Knot(prefix)
    assert os.path.isdir(download_dir) == True

    assert knot.path == extract_dir + "/KnotMesh.ply"
    assert os.path.isfile(knot.path) == True

    assert knot.prefix == prefix
    assert knot.data_root == data_root
    assert knot.download_dir == download_dir
    assert knot.extract_dir == extract_dir

    shutil.rmtree(download_dir, ignore_errors=True)
    shutil.rmtree(extract_dir, ignore_errors=True)


def test_juneau():
    prefix = "O3DTestJuneau"
    data_root = os.path.join(Path.home(), "open3d_data")
    download_dir = os.path.join(data_root, "download", prefix)
    extract_dir = os.path.join(data_root, "extract", prefix)

    shutil.rmtree(download_dir, ignore_errors=True)
    shutil.rmtree(extract_dir, ignore_errors=True)

    juneau = o3d.data.Juneau(prefix)
    assert os.path.isdir(download_dir) == True

    assert juneau.path == extract_dir + "/JuneauImage.jpg"
    assert os.path.isfile(juneau.path) == True

    assert juneau.prefix == prefix
    assert juneau.data_root == data_root
    assert juneau.download_dir == download_dir
    assert juneau.extract_dir == extract_dir

    shutil.rmtree(download_dir, ignore_errors=True)
    shutil.rmtree(extract_dir, ignore_errors=True)
