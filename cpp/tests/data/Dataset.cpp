// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018-2021 www.open3d.org
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// ----------------------------------------------------------------------------

#include "open3d/data/Dataset.h"

#include "open3d/t/io/PointCloudIO.h"
#include "open3d/utility/FileSystem.h"
#include "open3d/utility/Helper.h"
#include "open3d/utility/Logging.h"
#include "tests/Tests.h"

namespace open3d {
namespace tests {

TEST(Dataset, DatasetBase) {
    data::Dataset ds("some_prefix");
    EXPECT_EQ(ds.GetDataRoot(),
              utility::filesystem::GetHomeDirectory() + "/open3d_data");

    data::Dataset ds_custom("some_prefix", "/my/custom/data_root");
    EXPECT_EQ(ds_custom.GetPrefix(), "some_prefix");
    EXPECT_EQ(ds_custom.GetDataRoot(), "/my/custom/data_root");
    EXPECT_EQ(ds_custom.GetDownloadDir(),
              "/my/custom/data_root/download/some_prefix");
    EXPECT_EQ(ds_custom.GetExtractDir(),
              "/my/custom/data_root/extract/some_prefix");
}

TEST(Dataset, SingleDownloadDataset) {
    const std::string prefix = "SingleDownloadDataset";
    const std::string data_root =
            utility::filesystem::GetHomeDirectory() + "/open3d_data";
    const std::string download_dir = data_root + "/download/" + prefix;
    const std::string extract_dir = data_root + "/extract/" + prefix;

    const std::vector<std::string> url_mirrors = {
            "https://github.com/isl-org/open3d_downloads/releases/download/"
            "20220201-data/BunnyMesh.ply"};
    const std::string md5 = "568f871d1a221ba6627569f1e6f9a3f2";

    data::SingleDownloadDataset single_download_dataset(
            prefix, url_mirrors, md5,
            /*no_extact*/ true, data_root);

    EXPECT_TRUE(
            utility::filesystem::FileExists(download_dir + "/BunnyMesh.ply"));
    EXPECT_TRUE(
            utility::filesystem::FileExists(extract_dir + "/BunnyMesh.ply"));

    EXPECT_EQ(single_download_dataset.GetPrefix(), prefix);
    EXPECT_EQ(single_download_dataset.GetDataRoot(), data_root);
    EXPECT_EQ(single_download_dataset.GetDownloadDir(), download_dir);
    EXPECT_EQ(single_download_dataset.GetExtractDir(), extract_dir);
}

TEST(Dataset, DemoICPPointClouds) {
    const std::string prefix = "DemoICPPointClouds";
    const std::string data_root =
            utility::filesystem::GetHomeDirectory() + "/open3d_data";
    const std::string download_dir = data_root + "/download/" + prefix;
    const std::string extract_dir = data_root + "/extract/" + prefix;

    data::DemoICPPointClouds demo_icp;
    EXPECT_TRUE(utility::filesystem::DirectoryExists(download_dir));

    const std::vector<std::string> paths = {extract_dir + "/cloud_bin_0.pcd",
                                            extract_dir + "/cloud_bin_1.pcd",
                                            extract_dir + "/cloud_bin_2.pcd"};
    EXPECT_EQ(demo_icp.GetPaths(), paths);
    for (size_t i = 0; i < paths.size(); ++i) {
        EXPECT_EQ(demo_icp.GetPaths(i), paths[i]);
        EXPECT_TRUE(utility::filesystem::FileExists(demo_icp.GetPaths(i)));
    }

    EXPECT_EQ(demo_icp.GetTransformationLogPath(), extract_dir + "/init.log");
    EXPECT_TRUE(utility::filesystem::FileExists(
            demo_icp.GetTransformationLogPath()));

    EXPECT_EQ(demo_icp.GetPrefix(), prefix);
    EXPECT_EQ(demo_icp.GetDataRoot(), data_root);
    EXPECT_EQ(demo_icp.GetDownloadDir(), download_dir);
    EXPECT_EQ(demo_icp.GetExtractDir(), extract_dir);
}

TEST(Dataset, DemoColoredICPPointClouds) {
    const std::string prefix = "DemoColoredICPPointClouds";
    const std::string data_root =
            utility::filesystem::GetHomeDirectory() + "/open3d_data";
    const std::string download_dir = data_root + "/download/" + prefix;
    const std::string extract_dir = data_root + "/extract/" + prefix;

    data::DemoColoredICPPointClouds dataset;
    EXPECT_TRUE(utility::filesystem::DirectoryExists(download_dir));

    const std::vector<std::string> paths = {extract_dir + "/frag_115.ply",
                                            extract_dir + "/frag_116.ply"};
    EXPECT_EQ(dataset.GetPaths(), paths);
    for (size_t i = 0; i < paths.size(); ++i) {
        EXPECT_EQ(dataset.GetPaths(i), paths[i]);
        EXPECT_TRUE(utility::filesystem::FileExists(dataset.GetPaths(i)));
    }

    EXPECT_EQ(dataset.GetPrefix(), prefix);
    EXPECT_EQ(dataset.GetDataRoot(), data_root);
    EXPECT_EQ(dataset.GetDownloadDir(), download_dir);
    EXPECT_EQ(dataset.GetExtractDir(), extract_dir);
}

TEST(Dataset, DemoCropPointCloud) {
    const std::string prefix = "DemoCropPointCloud";
    const std::string data_root =
            utility::filesystem::GetHomeDirectory() + "/open3d_data";
    const std::string download_dir = data_root + "/download/" + prefix;
    const std::string extract_dir = data_root + "/extract/" + prefix;

    data::DemoCropPointCloud dataset;
    EXPECT_TRUE(utility::filesystem::DirectoryExists(download_dir));

    EXPECT_EQ(dataset.GetPointCloudPath(), extract_dir + "/fragment.ply");
    EXPECT_TRUE(utility::filesystem::FileExists(dataset.GetPointCloudPath()));
    EXPECT_EQ(dataset.GetCroppedJSONPath(), extract_dir + "/cropped.json");
    EXPECT_TRUE(utility::filesystem::FileExists(dataset.GetCroppedJSONPath()));

    EXPECT_EQ(dataset.GetPrefix(), prefix);
    EXPECT_EQ(dataset.GetDataRoot(), data_root);
    EXPECT_EQ(dataset.GetDownloadDir(), download_dir);
    EXPECT_EQ(dataset.GetExtractDir(), extract_dir);
}

TEST(Dataset, DemoFeatureMatchingPointClouds) {
    const std::string prefix = "DemoFeatureMatchingPointClouds";
    const std::string data_root =
            utility::filesystem::GetHomeDirectory() + "/open3d_data";
    const std::string download_dir = data_root + "/download/" + prefix;
    const std::string extract_dir = data_root + "/extract/" + prefix;

    data::DemoFeatureMatchingPointClouds dataset;
    EXPECT_TRUE(utility::filesystem::DirectoryExists(download_dir));

    const std::vector<std::string> point_cloud_paths = {
            extract_dir + "/cloud_bin_0.pcd", extract_dir + "/cloud_bin_1.pcd"};
    EXPECT_EQ(dataset.GetPointCloudPaths(), point_cloud_paths);

    const std::vector<std::string> fpfh_feature_paths = {
            extract_dir + "/cloud_bin_0.fpfh.bin",
            extract_dir + "/cloud_bin_1.fpfh.bin"};
    EXPECT_EQ(dataset.GetFPFHFeaturePaths(), fpfh_feature_paths);

    const std::vector<std::string> l32d_feature_paths = {
            extract_dir + "/cloud_bin_0.d32.bin",
            extract_dir + "/cloud_bin_1.d32.bin"};
    EXPECT_EQ(dataset.GetL32DFeaturePaths(), l32d_feature_paths);

    EXPECT_EQ(dataset.GetPrefix(), prefix);
    EXPECT_EQ(dataset.GetDataRoot(), data_root);
    EXPECT_EQ(dataset.GetDownloadDir(), download_dir);
    EXPECT_EQ(dataset.GetExtractDir(), extract_dir);
}

TEST(Dataset, DemoPoseGraphOptimization) {
    const std::string prefix = "DemoPoseGraphOptimization";
    const std::string data_root =
            utility::filesystem::GetHomeDirectory() + "/open3d_data";
    const std::string download_dir = data_root + "/download/" + prefix;
    const std::string extract_dir = data_root + "/extract/" + prefix;

    data::DemoPoseGraphOptimization dataset;
    EXPECT_TRUE(utility::filesystem::DirectoryExists(download_dir));

    EXPECT_EQ(dataset.GetPoseGraphFragmentPath(),
              extract_dir + "/pose_graph_example_fragment.json");
    EXPECT_EQ(dataset.GetPoseGraphGlobalPath(),
              extract_dir + "/pose_graph_example_global.json");

    EXPECT_EQ(dataset.GetPrefix(), prefix);
    EXPECT_EQ(dataset.GetDataRoot(), data_root);
    EXPECT_EQ(dataset.GetDownloadDir(), download_dir);
    EXPECT_EQ(dataset.GetExtractDir(), extract_dir);
}

TEST(Dataset, PCDPointCloud) {
    const std::string prefix = "PCDPointCloud";
    const std::string data_root =
            utility::filesystem::GetHomeDirectory() + "/open3d_data";
    const std::string download_dir = data_root + "/download/" + prefix;
    const std::string extract_dir = data_root + "/extract/" + prefix;

    data::PCDPointCloud dataset;
    EXPECT_TRUE(utility::filesystem::DirectoryExists(download_dir));

    EXPECT_EQ(dataset.GetPath(), extract_dir + "/fragment.pcd");
    EXPECT_TRUE(utility::filesystem::FileExists(dataset.GetPath()));

    EXPECT_EQ(dataset.GetPrefix(), prefix);
    EXPECT_EQ(dataset.GetDataRoot(), data_root);
    EXPECT_EQ(dataset.GetDownloadDir(), download_dir);
    EXPECT_EQ(dataset.GetExtractDir(), extract_dir);
}

TEST(Dataset, PLYPointCloud) {
    const std::string prefix = "PLYPointCloud";
    const std::string data_root =
            utility::filesystem::GetHomeDirectory() + "/open3d_data";
    const std::string download_dir = data_root + "/download/" + prefix;
    const std::string extract_dir = data_root + "/extract/" + prefix;

    data::PLYPointCloud dataset;
    EXPECT_TRUE(utility::filesystem::DirectoryExists(download_dir));

    EXPECT_EQ(dataset.GetPath(), extract_dir + "/fragment.ply");
    EXPECT_TRUE(utility::filesystem::FileExists(dataset.GetPath()));

    EXPECT_EQ(dataset.GetPrefix(), prefix);
    EXPECT_EQ(dataset.GetDataRoot(), data_root);
    EXPECT_EQ(dataset.GetDownloadDir(), download_dir);
    EXPECT_EQ(dataset.GetExtractDir(), extract_dir);
}

TEST(Dataset, PTSPointCloud) {
    const std::string prefix = "PTSPointCloud";
    const std::string data_root =
            utility::filesystem::GetHomeDirectory() + "/open3d_data";
    const std::string download_dir = data_root + "/download/" + prefix;
    const std::string extract_dir = data_root + "/extract/" + prefix;

    data::PTSPointCloud dataset;
    EXPECT_TRUE(utility::filesystem::DirectoryExists(download_dir));

    EXPECT_EQ(dataset.GetPath(), extract_dir + "/point_cloud_sample1.pts");
    EXPECT_TRUE(utility::filesystem::FileExists(dataset.GetPath()));

    EXPECT_EQ(dataset.GetPrefix(), prefix);
    EXPECT_EQ(dataset.GetDataRoot(), data_root);
    EXPECT_EQ(dataset.GetDownloadDir(), download_dir);
    EXPECT_EQ(dataset.GetExtractDir(), extract_dir);
}

TEST(Dataset, SampleNYURGBDImage) {
    const std::string prefix = "SampleNYURGBDImage";
    const std::string data_root =
            utility::filesystem::GetHomeDirectory() + "/open3d_data";
    const std::string download_dir = data_root + "/download/" + prefix;
    const std::string extract_dir = data_root + "/extract/" + prefix;

    data::SampleNYURGBDImage dataset;
    EXPECT_TRUE(utility::filesystem::DirectoryExists(download_dir));

    EXPECT_EQ(dataset.GetColorPath(), extract_dir + "/NYU_color.ppm");
    EXPECT_TRUE(utility::filesystem::FileExists(dataset.GetColorPath()));
    EXPECT_EQ(dataset.GetDepthPath(), extract_dir + "/NYU_depth.pgm");
    EXPECT_TRUE(utility::filesystem::FileExists(dataset.GetDepthPath()));

    EXPECT_EQ(dataset.GetPrefix(), prefix);
    EXPECT_EQ(dataset.GetDataRoot(), data_root);
    EXPECT_EQ(dataset.GetDownloadDir(), download_dir);
    EXPECT_EQ(dataset.GetExtractDir(), extract_dir);
}

TEST(Dataset, SampleSUNRGBDImage) {
    const std::string prefix = "SampleSUNRGBDImage";
    const std::string data_root =
            utility::filesystem::GetHomeDirectory() + "/open3d_data";
    const std::string download_dir = data_root + "/download/" + prefix;
    const std::string extract_dir = data_root + "/extract/" + prefix;

    data::SampleSUNRGBDImage dataset;
    EXPECT_TRUE(utility::filesystem::DirectoryExists(download_dir));

    EXPECT_EQ(dataset.GetColorPath(), extract_dir + "/SUN_color.jpg");
    EXPECT_TRUE(utility::filesystem::FileExists(dataset.GetColorPath()));
    EXPECT_EQ(dataset.GetDepthPath(), extract_dir + "/SUN_depth.png");
    EXPECT_TRUE(utility::filesystem::FileExists(dataset.GetDepthPath()));

    EXPECT_EQ(dataset.GetPrefix(), prefix);
    EXPECT_EQ(dataset.GetDataRoot(), data_root);
    EXPECT_EQ(dataset.GetDownloadDir(), download_dir);
    EXPECT_EQ(dataset.GetExtractDir(), extract_dir);
}

TEST(Dataset, SampleTUMRGBDImage) {
    const std::string prefix = "SampleTUMRGBDImage";
    const std::string data_root =
            utility::filesystem::GetHomeDirectory() + "/open3d_data";
    const std::string download_dir = data_root + "/download/" + prefix;
    const std::string extract_dir = data_root + "/extract/" + prefix;

    data::SampleTUMRGBDImage dataset;
    EXPECT_TRUE(utility::filesystem::DirectoryExists(download_dir));

    EXPECT_EQ(dataset.GetColorPath(), extract_dir + "/TUM_color.png");
    EXPECT_TRUE(utility::filesystem::FileExists(dataset.GetColorPath()));
    EXPECT_EQ(dataset.GetDepthPath(), extract_dir + "/TUM_depth.png");
    EXPECT_TRUE(utility::filesystem::FileExists(dataset.GetDepthPath()));

    EXPECT_EQ(dataset.GetPrefix(), prefix);
    EXPECT_EQ(dataset.GetDataRoot(), data_root);
    EXPECT_EQ(dataset.GetDownloadDir(), download_dir);
    EXPECT_EQ(dataset.GetExtractDir(), extract_dir);
}

TEST(Dataset, SampleRedwoodRGBDImages) {
    const std::string prefix = "SampleRedwoodRGBDImages";
    const std::string data_root =
            utility::filesystem::GetHomeDirectory() + "/open3d_data";
    const std::string download_dir = data_root + "/download/" + prefix;
    const std::string extract_dir = data_root + "/extract/" + prefix;

    data::SampleRedwoodRGBDImages dataset;
    EXPECT_TRUE(utility::filesystem::DirectoryExists(download_dir));

    const std::vector<std::string> color_paths = {
            extract_dir + "/color/00000.jpg", extract_dir + "/color/00001.jpg",
            extract_dir + "/color/00002.jpg", extract_dir + "/color/00003.jpg",
            extract_dir + "/color/00004.jpg"};
    EXPECT_EQ(dataset.GetColorPaths(), color_paths);

    const std::vector<std::string> depth_paths = {
            extract_dir + "/depth/00000.png", extract_dir + "/depth/00001.png",
            extract_dir + "/depth/00002.png", extract_dir + "/depth/00003.png",
            extract_dir + "/depth/00004.png"};
    EXPECT_EQ(dataset.GetDepthPaths(), depth_paths);
    for (size_t i = 0; i < color_paths.size(); ++i) {
        EXPECT_TRUE(
                utility::filesystem::FileExists(dataset.GetColorPaths()[i]));
        EXPECT_TRUE(
                utility::filesystem::FileExists(dataset.GetDepthPaths()[i]));
    }

    EXPECT_EQ(dataset.GetTrajectoryLogPath(), extract_dir + "/trajectory.log");
    EXPECT_TRUE(
            utility::filesystem::FileExists(dataset.GetTrajectoryLogPath()));

    EXPECT_EQ(dataset.GetOdometryLogPath(), extract_dir + "/odometry.log");
    EXPECT_TRUE(utility::filesystem::FileExists(dataset.GetOdometryLogPath()));

    EXPECT_EQ(dataset.GetRGBDMatchPath(), extract_dir + "/rgbd.match");
    EXPECT_TRUE(utility::filesystem::FileExists(dataset.GetRGBDMatchPath()));

    EXPECT_EQ(dataset.GetReconstructionPath(),
              extract_dir + "/example_tsdf_pcd.ply");
    EXPECT_TRUE(
            utility::filesystem::FileExists(dataset.GetReconstructionPath()));

    EXPECT_EQ(dataset.GetPrefix(), prefix);
    EXPECT_EQ(dataset.GetDataRoot(), data_root);
    EXPECT_EQ(dataset.GetDownloadDir(), download_dir);
    EXPECT_EQ(dataset.GetExtractDir(), extract_dir);
}

TEST(Dataset, SampleFountainRGBDImages) {
    const std::string prefix = "SampleFountainRGBDImages";
    const std::string data_root =
            utility::filesystem::GetHomeDirectory() + "/open3d_data";
    const std::string download_dir = data_root + "/download/" + prefix;
    const std::string extract_dir = data_root + "/extract/" + prefix;

    data::SampleFountainRGBDImages dataset;
    EXPECT_TRUE(utility::filesystem::DirectoryExists(download_dir));

    const std::vector<std::string> color_paths = {
            extract_dir + "/image/0000010-000001228920.jpg",
            extract_dir + "/image/0000031-000004096400.jpg",
            extract_dir + "/image/0000044-000005871507.jpg",
            extract_dir + "/image/0000064-000008602440.jpg",
            extract_dir + "/image/0000110-000014883587.jpg",
            extract_dir + "/image/0000156-000021164733.jpg",
            extract_dir + "/image/0000200-000027172787.jpg",
            extract_dir + "/image/0000215-000029220987.jpg",
            extract_dir + "/image/0000255-000034682853.jpg",
            extract_dir + "/image/0000299-000040690907.jpg",
            extract_dir + "/image/0000331-000045060400.jpg",
            extract_dir + "/image/0000368-000050112627.jpg",
            extract_dir + "/image/0000412-000056120680.jpg",
            extract_dir + "/image/0000429-000058441973.jpg",
            extract_dir + "/image/0000474-000064586573.jpg",
            extract_dir + "/image/0000487-000066361680.jpg",
            extract_dir + "/image/0000526-000071687000.jpg",
            extract_dir + "/image/0000549-000074827573.jpg",
            extract_dir + "/image/0000582-000079333613.jpg",
            extract_dir + "/image/0000630-000085887853.jpg",
            extract_dir + "/image/0000655-000089301520.jpg",
            extract_dir + "/image/0000703-000095855760.jpg",
            extract_dir + "/image/0000722-000098450147.jpg",
            extract_dir + "/image/0000771-000105140933.jpg",
            extract_dir + "/image/0000792-000108008413.jpg",
            extract_dir + "/image/0000818-000111558627.jpg",
            extract_dir + "/image/0000849-000115791573.jpg",
            extract_dir + "/image/0000883-000120434160.jpg",
            extract_dir + "/image/0000896-000122209267.jpg",
            extract_dir + "/image/0000935-000127534587.jpg",
            extract_dir + "/image/0000985-000134361920.jpg",
            extract_dir + "/image/0001028-000140233427.jpg",
            extract_dir + "/image/0001061-000144739467.jpg"};
    EXPECT_EQ(dataset.GetColorPaths(), color_paths);

    const std::vector<std::string> depth_paths = {
            extract_dir + "/depth/0000038-000001234662.png",
            extract_dir + "/depth/0000124-000004104418.png",
            extract_dir + "/depth/0000177-000005872988.png",
            extract_dir + "/depth/0000259-000008609267.png",
            extract_dir + "/depth/0000447-000014882686.png",
            extract_dir + "/depth/0000635-000021156105.png",
            extract_dir + "/depth/0000815-000027162570.png",
            extract_dir + "/depth/0000877-000029231463.png",
            extract_dir + "/depth/0001040-000034670651.png",
            extract_dir + "/depth/0001220-000040677116.png",
            extract_dir + "/depth/0001351-000045048488.png",
            extract_dir + "/depth/0001503-000050120614.png",
            extract_dir + "/depth/0001683-000056127079.png",
            extract_dir + "/depth/0001752-000058429557.png",
            extract_dir + "/depth/0001937-000064602868.png",
            extract_dir + "/depth/0001990-000066371438.png",
            extract_dir + "/depth/0002149-000071677149.png",
            extract_dir + "/depth/0002243-000074813859.png",
            extract_dir + "/depth/0002378-000079318707.png",
            extract_dir + "/depth/0002575-000085892450.png",
            extract_dir + "/depth/0002677-000089296113.png",
            extract_dir + "/depth/0002874-000095869855.png",
            extract_dir + "/depth/0002951-000098439288.png",
            extract_dir + "/depth/0003152-000105146507.png",
            extract_dir + "/depth/0003238-000108016262.png",
            extract_dir + "/depth/0003344-000111553403.png",
            extract_dir + "/depth/0003471-000115791298.png",
            extract_dir + "/depth/0003610-000120429623.png",
            extract_dir + "/depth/0003663-000122198194.png",
            extract_dir + "/depth/0003823-000127537274.png",
            extract_dir + "/depth/0004028-000134377970.png",
            extract_dir + "/depth/0004203-000140217589.png",
            extract_dir + "/depth/0004339-000144755807.png"};
    EXPECT_EQ(dataset.GetDepthPaths(), depth_paths);

    for (auto& color_path : dataset.GetColorPaths()) {
        EXPECT_TRUE(utility::filesystem::FileExists(color_path));
    }
    for (auto& depth_path : dataset.GetDepthPaths()) {
        EXPECT_TRUE(utility::filesystem::FileExists(depth_path));
    }

    EXPECT_EQ(dataset.GetKeyframePosesLogPath(),
              extract_dir + "/scene/key.log");
    EXPECT_TRUE(
            utility::filesystem::FileExists(dataset.GetKeyframePosesLogPath()));

    EXPECT_EQ(dataset.GetReconstructionPath(),
              extract_dir + "/scene/integrated.ply");
    EXPECT_TRUE(
            utility::filesystem::FileExists(dataset.GetReconstructionPath()));

    EXPECT_EQ(dataset.GetPrefix(), prefix);
    EXPECT_EQ(dataset.GetDataRoot(), data_root);
    EXPECT_EQ(dataset.GetDownloadDir(), download_dir);
    EXPECT_EQ(dataset.GetExtractDir(), extract_dir);
}

TEST(Dataset, SampleL515Bag) {
    const std::string prefix = "SampleL515Bag";
    const std::string data_root =
            utility::filesystem::GetHomeDirectory() + "/open3d_data";
    const std::string download_dir = data_root + "/download/" + prefix;
    const std::string extract_dir = data_root + "/extract/" + prefix;

    data::SampleL515Bag dataset;
    EXPECT_TRUE(utility::filesystem::DirectoryExists(download_dir));

    EXPECT_EQ(dataset.GetPath(), extract_dir + "/L515_test_s.bag");
    EXPECT_TRUE(utility::filesystem::FileExists(dataset.GetPath()));

    EXPECT_EQ(dataset.GetPrefix(), prefix);
    EXPECT_EQ(dataset.GetDataRoot(), data_root);
    EXPECT_EQ(dataset.GetDownloadDir(), download_dir);
    EXPECT_EQ(dataset.GetExtractDir(), extract_dir);
}

TEST(Dataset, EaglePointCloud) {
    const std::string prefix = "EaglePointCloud";
    const std::string data_root =
            utility::filesystem::GetHomeDirectory() + "/open3d_data";
    const std::string download_dir = data_root + "/download/" + prefix;
    const std::string extract_dir = data_root + "/extract/" + prefix;

    data::EaglePointCloud dataset;
    EXPECT_TRUE(utility::filesystem::DirectoryExists(download_dir));

    EXPECT_EQ(dataset.GetPath(), extract_dir + "/EaglePointCloud.ply");
    EXPECT_TRUE(utility::filesystem::FileExists(dataset.GetPath()));

    EXPECT_EQ(dataset.GetPrefix(), prefix);
    EXPECT_EQ(dataset.GetDataRoot(), data_root);
    EXPECT_EQ(dataset.GetDownloadDir(), download_dir);
    EXPECT_EQ(dataset.GetExtractDir(), extract_dir);
}

TEST(Dataset, ArmadilloMesh) {
    const std::string prefix = "ArmadilloMesh";
    const std::string data_root =
            utility::filesystem::GetHomeDirectory() + "/open3d_data";
    const std::string download_dir = data_root + "/download/" + prefix;
    const std::string extract_dir = data_root + "/extract/" + prefix;

    data::ArmadilloMesh dataset;
    EXPECT_TRUE(utility::filesystem::DirectoryExists(download_dir));

    EXPECT_EQ(dataset.GetPath(), extract_dir + "/ArmadilloMesh.ply");
    EXPECT_TRUE(utility::filesystem::FileExists(dataset.GetPath()));

    EXPECT_EQ(dataset.GetPrefix(), prefix);
    EXPECT_EQ(dataset.GetDataRoot(), data_root);
    EXPECT_EQ(dataset.GetDownloadDir(), download_dir);
    EXPECT_EQ(dataset.GetExtractDir(), extract_dir);
}

TEST(Dataset, BunnyMesh) {
    const std::string prefix = "BunnyMesh";
    const std::string data_root =
            utility::filesystem::GetHomeDirectory() + "/open3d_data";
    const std::string download_dir = data_root + "/download/" + prefix;
    const std::string extract_dir = data_root + "/extract/" + prefix;

    data::BunnyMesh dataset;
    EXPECT_TRUE(utility::filesystem::DirectoryExists(download_dir));

    EXPECT_EQ(dataset.GetPath(), extract_dir + "/BunnyMesh.ply");
    EXPECT_TRUE(utility::filesystem::FileExists(dataset.GetPath()));

    EXPECT_EQ(dataset.GetPrefix(), prefix);
    EXPECT_EQ(dataset.GetDataRoot(), data_root);
    EXPECT_EQ(dataset.GetDownloadDir(), download_dir);
    EXPECT_EQ(dataset.GetExtractDir(), extract_dir);
}

TEST(Dataset, KnotMesh) {
    const std::string prefix = "KnotMesh";
    const std::string data_root =
            utility::filesystem::GetHomeDirectory() + "/open3d_data";
    const std::string download_dir = data_root + "/download/" + prefix;
    const std::string extract_dir = data_root + "/extract/" + prefix;

    data::KnotMesh dataset;
    EXPECT_TRUE(utility::filesystem::DirectoryExists(download_dir));

    EXPECT_EQ(dataset.GetPath(), extract_dir + "/KnotMesh.ply");
    EXPECT_TRUE(utility::filesystem::FileExists(dataset.GetPath()));

    EXPECT_EQ(dataset.GetPrefix(), prefix);
    EXPECT_EQ(dataset.GetDataRoot(), data_root);
    EXPECT_EQ(dataset.GetDownloadDir(), download_dir);
    EXPECT_EQ(dataset.GetExtractDir(), extract_dir);
}

TEST(Dataset, MonkeyModel) {
    const std::string prefix = "MonkeyModel";
    const std::string data_root =
            utility::filesystem::GetHomeDirectory() + "/open3d_data";
    const std::string download_dir = data_root + "/download/" + prefix;
    const std::string extract_dir = data_root + "/extract/" + prefix;

    data::MonkeyModel dataset;
    EXPECT_TRUE(utility::filesystem::DirectoryExists(download_dir));

    std::unordered_map<std::string, std::string> map_filename_to_path = {
            {"albedo", extract_dir + "/albedo.png"},
            {"ao", extract_dir + "/ao.png"},
            {"metallic", extract_dir + "/metallic.png"},
            {"monkey_material", extract_dir + "/monkey.mtl"},
            {"monkey_model", extract_dir + "/monkey.obj"},
            {"monkey_solid_material", extract_dir + "/monkey_solid.mtl"},
            {"monkey_solid_model", extract_dir + "/monkey_solid.obj"},
            {"normal", extract_dir + "/normal.png"},
            {"roughness", extract_dir + "/roughness.png"}};

    for (auto file_name : dataset.GetPathMap()) {
        EXPECT_EQ(map_filename_to_path.at(file_name.first), file_name.second);
        EXPECT_EQ(dataset.GetPath(file_name.first), file_name.second);
        EXPECT_TRUE(utility::filesystem::FileExists(file_name.second));
    }

    EXPECT_EQ(dataset.GetPrefix(), prefix);
    EXPECT_EQ(dataset.GetDataRoot(), data_root);
    EXPECT_EQ(dataset.GetDownloadDir(), download_dir);
    EXPECT_EQ(dataset.GetExtractDir(), extract_dir);
}

TEST(Dataset, SwordModel) {
    const std::string prefix = "SwordModel";
    const std::string data_root =
            utility::filesystem::GetHomeDirectory() + "/open3d_data";
    const std::string download_dir = data_root + "/download/" + prefix;
    const std::string extract_dir = data_root + "/extract/" + prefix;

    data::SwordModel dataset;
    EXPECT_TRUE(utility::filesystem::DirectoryExists(download_dir));

    std::unordered_map<std::string, std::string> map_filename_to_path = {
            {"sword_material", extract_dir + "/UV.mtl"},
            {"sword_model", extract_dir + "/UV.obj"},
            {"base_color", extract_dir + "/UV_blinn1SG_BaseColor.png"},
            {"metallic", extract_dir + "/UV_blinn1SG_Metallic.png"},
            {"normal", extract_dir + "/UV_blinn1SG_Normal.png"},
            {"roughness", extract_dir + "/UV_blinn1SG_Roughness.png"}};

    for (auto file_name : dataset.GetPathMap()) {
        EXPECT_EQ(map_filename_to_path.at(file_name.first), file_name.second);
        EXPECT_EQ(dataset.GetPath(file_name.first), file_name.second);
        EXPECT_TRUE(utility::filesystem::FileExists(file_name.second));
    }

    EXPECT_EQ(dataset.GetPrefix(), prefix);
    EXPECT_EQ(dataset.GetDataRoot(), data_root);
    EXPECT_EQ(dataset.GetDownloadDir(), download_dir);
    EXPECT_EQ(dataset.GetExtractDir(), extract_dir);
}

TEST(Dataset, CrateModel) {
    const std::string prefix = "CrateModel";
    const std::string data_root =
            utility::filesystem::GetHomeDirectory() + "/open3d_data";
    const std::string download_dir = data_root + "/download/" + prefix;
    const std::string extract_dir = data_root + "/extract/" + prefix;

    data::CrateModel dataset;
    EXPECT_TRUE(utility::filesystem::DirectoryExists(download_dir));

    std::unordered_map<std::string, std::string> map_filename_to_path = {
            {"crate_material", extract_dir + "/crate.mtl"},
            {"crate_model", extract_dir + "/crate.obj"},
            {"texture_image", extract_dir + "/crate.jpg"}};

    for (auto file_name : dataset.GetPathMap()) {
        EXPECT_EQ(map_filename_to_path.at(file_name.first), file_name.second);
        EXPECT_EQ(dataset.GetPath(file_name.first), file_name.second);
        EXPECT_TRUE(utility::filesystem::FileExists(file_name.second));
    }

    EXPECT_EQ(dataset.GetPrefix(), prefix);
    EXPECT_EQ(dataset.GetDataRoot(), data_root);
    EXPECT_EQ(dataset.GetDownloadDir(), download_dir);
    EXPECT_EQ(dataset.GetExtractDir(), extract_dir);
}

TEST(Dataset, FlightHelmetModel) {
    const std::string prefix = "FlightHelmetModel";
    const std::string data_root =
            utility::filesystem::GetHomeDirectory() + "/open3d_data";
    const std::string download_dir = data_root + "/download/" + prefix;
    const std::string extract_dir = data_root + "/extract/" + prefix;

    data::FlightHelmetModel dataset;
    EXPECT_TRUE(utility::filesystem::DirectoryExists(download_dir));

    std::unordered_map<std::string, std::string> map_filename_to_path = {
            {"flight_helmet", extract_dir + "/FlightHelmet.gltf"},
            {"flight_helmet_bin", extract_dir + "/FlightHelmet.bin"},
            {"mat_glass_plastic_base",
             extract_dir +
                     "/FlightHelmet_Materials_GlassPlasticMat_BaseColor.png"},
            {"mat_glass_plastic_normal",
             extract_dir +
                     "/FlightHelmet_Materials_GlassPlasticMat_Normal.png"},
            {"mat_glass_plastic_occlusion_rough_metal",
             extract_dir + "/FlightHelmet_Materials_GlassPlasticMat_"
                           "OcclusionRoughMetal.png"},
            {"mat_leather_parts_base",
             extract_dir +
                     "/FlightHelmet_Materials_LeatherPartsMat_BaseColor.png"},
            {"mat_leather_parts_normal",
             extract_dir +
                     "/FlightHelmet_Materials_LeatherPartsMat_Normal.png"},
            {"mat_leather_parts_occlusion_rough_metal",
             extract_dir + "/FlightHelmet_Materials_LeatherPartsMat_"
                           "OcclusionRoughMetal.png"},
            {"mat_lenses_base",
             extract_dir + "/FlightHelmet_Materials_LensesMat_BaseColor.png"},
            {"mat_lenses_normal",
             extract_dir + "/FlightHelmet_Materials_LensesMat_Normal.png"},
            {"mat_lenses_occlusion_rough_metal",
             extract_dir + "/FlightHelmet_Materials_LensesMat_"
                           "OcclusionRoughMetal.png"},
            {"mat_metal_parts_base",
             extract_dir +
                     "/FlightHelmet_Materials_MetalPartsMat_BaseColor.png"},
            {"mat_metal_parts_normal",
             extract_dir + "/FlightHelmet_Materials_MetalPartsMat_Normal.png"},
            {"mat_metal_parts_occlusion_rough_metal",
             extract_dir + "/FlightHelmet_Materials_MetalPartsMat_"
                           "OcclusionRoughMetal.png"},
            {"mat_rubber_wood_base",
             extract_dir +
                     "/FlightHelmet_Materials_RubberWoodMat_BaseColor.png"},
            {"mat_rubber_wood_normal",
             extract_dir + "/FlightHelmet_Materials_RubberWoodMat_Normal.png"},
            {"mat_rubber_wood_occlusion_rough_metal",
             extract_dir + "/FlightHelmet_Materials_RubberWoodMat_"
                           "OcclusionRoughMetal.png"}};

    for (auto file_name : dataset.GetPathMap()) {
        EXPECT_EQ(map_filename_to_path.at(file_name.first), file_name.second);
        EXPECT_EQ(dataset.GetPath(file_name.first), file_name.second);
        EXPECT_TRUE(utility::filesystem::FileExists(file_name.second));
    }

    EXPECT_EQ(dataset.GetPrefix(), prefix);
    EXPECT_EQ(dataset.GetDataRoot(), data_root);
    EXPECT_EQ(dataset.GetDownloadDir(), download_dir);
    EXPECT_EQ(dataset.GetExtractDir(), extract_dir);
}

TEST(Dataset, MetalTexture) {
    const std::string prefix = "MetalTexture";
    const std::string data_root =
            utility::filesystem::GetHomeDirectory() + "/open3d_data";
    const std::string download_dir = data_root + "/download/" + prefix;
    const std::string extract_dir = data_root + "/extract/" + prefix;

    data::MetalTexture dataset;
    EXPECT_TRUE(utility::filesystem::DirectoryExists(download_dir));

    std::unordered_map<std::string, std::string> map_filename_to_path = {
            {"albedo", extract_dir + "/Metal008_Color.jpg"},
            {"normal", extract_dir + "/Metal008_NormalDX.jpg"},
            {"roughness", extract_dir + "/Metal008_Roughness.jpg"},
            {"metallic", extract_dir + "/Metal008_Metalness.jpg"}};

    for (auto file_name : dataset.GetPathMap()) {
        EXPECT_EQ(map_filename_to_path.at(file_name.first), file_name.second);
        EXPECT_TRUE(utility::filesystem::FileExists(file_name.second));
    }

    EXPECT_EQ(dataset.GetAlbedoTexturePath(),
              map_filename_to_path.at("albedo"));
    EXPECT_EQ(dataset.GetNormalTexturePath(),
              map_filename_to_path.at("normal"));
    EXPECT_EQ(dataset.GetRoughnessTexturePath(),
              map_filename_to_path.at("roughness"));
    EXPECT_EQ(dataset.GetMetallicTexturePath(),
              map_filename_to_path.at("metallic"));

    EXPECT_EQ(dataset.GetPrefix(), prefix);
    EXPECT_EQ(dataset.GetDataRoot(), data_root);
    EXPECT_EQ(dataset.GetDownloadDir(), download_dir);
    EXPECT_EQ(dataset.GetExtractDir(), extract_dir);
}

TEST(Dataset, PaintedPlasterTexture) {
    const std::string prefix = "PaintedPlasterTexture";
    const std::string data_root =
            utility::filesystem::GetHomeDirectory() + "/open3d_data";
    const std::string download_dir = data_root + "/download/" + prefix;
    const std::string extract_dir = data_root + "/extract/" + prefix;

    data::PaintedPlasterTexture dataset;
    EXPECT_TRUE(utility::filesystem::DirectoryExists(download_dir));

    std::unordered_map<std::string, std::string> map_filename_to_path = {
            {"albedo", extract_dir + "/PaintedPlaster017_Color.jpg"},
            {"normal", extract_dir + "/PaintedPlaster017_NormalDX.jpg"},
            {"roughness", extract_dir + "/noiseTexture.png"}};

    for (auto file_name : dataset.GetPathMap()) {
        EXPECT_EQ(map_filename_to_path.at(file_name.first), file_name.second);
        EXPECT_TRUE(utility::filesystem::FileExists(file_name.second));
    }

    EXPECT_EQ(dataset.GetAlbedoTexturePath(),
              map_filename_to_path.at("albedo"));
    EXPECT_EQ(dataset.GetNormalTexturePath(),
              map_filename_to_path.at("normal"));
    EXPECT_EQ(dataset.GetRoughnessTexturePath(),
              map_filename_to_path.at("roughness"));

    EXPECT_EQ(dataset.GetPrefix(), prefix);
    EXPECT_EQ(dataset.GetDataRoot(), data_root);
    EXPECT_EQ(dataset.GetDownloadDir(), download_dir);
    EXPECT_EQ(dataset.GetExtractDir(), extract_dir);
}

TEST(Dataset, TilesTexture) {
    const std::string prefix = "TilesTexture";
    const std::string data_root =
            utility::filesystem::GetHomeDirectory() + "/open3d_data";
    const std::string download_dir = data_root + "/download/" + prefix;
    const std::string extract_dir = data_root + "/extract/" + prefix;

    data::TilesTexture dataset;
    EXPECT_TRUE(utility::filesystem::DirectoryExists(download_dir));

    std::unordered_map<std::string, std::string> map_filename_to_path = {
            {"albedo", extract_dir + "/Tiles074_Color.jpg"},
            {"normal", extract_dir + "/Tiles074_NormalDX.jpg"},
            {"roughness", extract_dir + "/Tiles074_Roughness.jpg"}};

    for (auto file_name : dataset.GetPathMap()) {
        EXPECT_EQ(map_filename_to_path.at(file_name.first), file_name.second);
        EXPECT_TRUE(utility::filesystem::FileExists(file_name.second));
    }

    EXPECT_EQ(dataset.GetAlbedoTexturePath(),
              map_filename_to_path.at("albedo"));
    EXPECT_EQ(dataset.GetNormalTexturePath(),
              map_filename_to_path.at("normal"));
    EXPECT_EQ(dataset.GetRoughnessTexturePath(),
              map_filename_to_path.at("roughness"));

    EXPECT_EQ(dataset.GetPrefix(), prefix);
    EXPECT_EQ(dataset.GetDataRoot(), data_root);
    EXPECT_EQ(dataset.GetDownloadDir(), download_dir);
    EXPECT_EQ(dataset.GetExtractDir(), extract_dir);
}

TEST(Dataset, TerrazzoTexture) {
    const std::string prefix = "TerrazzoTexture";
    const std::string data_root =
            utility::filesystem::GetHomeDirectory() + "/open3d_data";
    const std::string download_dir = data_root + "/download/" + prefix;
    const std::string extract_dir = data_root + "/extract/" + prefix;

    data::TerrazzoTexture dataset;
    EXPECT_TRUE(utility::filesystem::DirectoryExists(download_dir));

    std::unordered_map<std::string, std::string> map_filename_to_path = {
            {"albedo", extract_dir + "/Terrazzo018_Color.jpg"},
            {"normal", extract_dir + "/Terrazzo018_NormalDX.jpg"},
            {"roughness", extract_dir + "/Terrazzo018_Roughness.jpg"}};

    for (auto file_name : dataset.GetPathMap()) {
        EXPECT_EQ(map_filename_to_path.at(file_name.first), file_name.second);
        EXPECT_TRUE(utility::filesystem::FileExists(file_name.second));
    }

    EXPECT_EQ(dataset.GetAlbedoTexturePath(),
              map_filename_to_path.at("albedo"));
    EXPECT_EQ(dataset.GetNormalTexturePath(),
              map_filename_to_path.at("normal"));
    EXPECT_EQ(dataset.GetRoughnessTexturePath(),
              map_filename_to_path.at("roughness"));

    EXPECT_EQ(dataset.GetPrefix(), prefix);
    EXPECT_EQ(dataset.GetDataRoot(), data_root);
    EXPECT_EQ(dataset.GetDownloadDir(), download_dir);
    EXPECT_EQ(dataset.GetExtractDir(), extract_dir);
}

TEST(Dataset, WoodTexture) {
    const std::string prefix = "WoodTexture";
    const std::string data_root =
            utility::filesystem::GetHomeDirectory() + "/open3d_data";
    const std::string download_dir = data_root + "/download/" + prefix;
    const std::string extract_dir = data_root + "/extract/" + prefix;

    data::WoodTexture dataset;
    EXPECT_TRUE(utility::filesystem::DirectoryExists(download_dir));

    std::unordered_map<std::string, std::string> map_filename_to_path = {
            {"albedo", extract_dir + "/Wood049_Color.jpg"},
            {"normal", extract_dir + "/Wood049_NormalDX.jpg"},
            {"roughness", extract_dir + "/Wood049_Roughness.jpg"}};

    for (auto file_name : dataset.GetPathMap()) {
        EXPECT_EQ(map_filename_to_path.at(file_name.first), file_name.second);
        EXPECT_TRUE(utility::filesystem::FileExists(file_name.second));
    }

    EXPECT_EQ(dataset.GetAlbedoTexturePath(),
              map_filename_to_path.at("albedo"));
    EXPECT_EQ(dataset.GetNormalTexturePath(),
              map_filename_to_path.at("normal"));
    EXPECT_EQ(dataset.GetRoughnessTexturePath(),
              map_filename_to_path.at("roughness"));

    EXPECT_EQ(dataset.GetPrefix(), prefix);
    EXPECT_EQ(dataset.GetDataRoot(), data_root);
    EXPECT_EQ(dataset.GetDownloadDir(), download_dir);
    EXPECT_EQ(dataset.GetExtractDir(), extract_dir);
}

TEST(Dataset, WoodFloorTexture) {
    const std::string prefix = "WoodFloorTexture";
    const std::string data_root =
            utility::filesystem::GetHomeDirectory() + "/open3d_data";
    const std::string download_dir = data_root + "/download/" + prefix;
    const std::string extract_dir = data_root + "/extract/" + prefix;

    data::WoodFloorTexture dataset;
    EXPECT_TRUE(utility::filesystem::DirectoryExists(download_dir));

    std::unordered_map<std::string, std::string> map_filename_to_path = {
            {"albedo", extract_dir + "/WoodFloor050_Color.jpg"},
            {"normal", extract_dir + "/WoodFloor050_NormalDX.jpg"},
            {"roughness", extract_dir + "/WoodFloor050_Roughness.jpg"}};

    for (auto file_name : dataset.GetPathMap()) {
        EXPECT_EQ(map_filename_to_path.at(file_name.first), file_name.second);
        EXPECT_TRUE(utility::filesystem::FileExists(file_name.second));
    }

    EXPECT_EQ(dataset.GetAlbedoTexturePath(),
              map_filename_to_path.at("albedo"));
    EXPECT_EQ(dataset.GetNormalTexturePath(),
              map_filename_to_path.at("normal"));
    EXPECT_EQ(dataset.GetRoughnessTexturePath(),
              map_filename_to_path.at("roughness"));

    EXPECT_EQ(dataset.GetPrefix(), prefix);
    EXPECT_EQ(dataset.GetDataRoot(), data_root);
    EXPECT_EQ(dataset.GetDownloadDir(), download_dir);
    EXPECT_EQ(dataset.GetExtractDir(), extract_dir);
}

TEST(Dataset, JuneauImage) {
    const std::string prefix = "JuneauImage";
    const std::string data_root =
            utility::filesystem::GetHomeDirectory() + "/open3d_data";
    const std::string download_dir = data_root + "/download/" + prefix;
    const std::string extract_dir = data_root + "/extract/" + prefix;

    data::JuneauImage dataset;
    EXPECT_TRUE(utility::filesystem::DirectoryExists(download_dir));

    EXPECT_EQ(dataset.GetPath(), extract_dir + "/JuneauImage.jpg");
    EXPECT_TRUE(utility::filesystem::FileExists(dataset.GetPath()));

    EXPECT_EQ(dataset.GetPrefix(), prefix);
    EXPECT_EQ(dataset.GetDataRoot(), data_root);
    EXPECT_EQ(dataset.GetDownloadDir(), download_dir);
    EXPECT_EQ(dataset.GetExtractDir(), extract_dir);
}

TEST(Dataset, DISABLED_RedwoodLivingRoomPointClouds) {
    const std::string prefix = "LivingRoomPointClouds";
    const std::string data_root =
            utility::filesystem::GetHomeDirectory() + "/open3d_data";
    const std::string download_dir = data_root + "/download/" + prefix;
    const std::string extract_dir = data_root + "/extract/" + prefix;

    data::LivingRoomPointClouds dataset;
    EXPECT_TRUE(utility::filesystem::DirectoryExists(download_dir));

    std::vector<std::string> paths;
    paths.reserve(57);
    for (int i = 0; i < 57; ++i) {
        paths.push_back(extract_dir + "/cloud_bin_" + std::to_string(i) +
                        ".ply");
    }
    EXPECT_EQ(dataset.GetPaths(), paths);
    for (size_t i = 0; i < paths.size(); ++i) {
        EXPECT_EQ(dataset.GetPaths(i), paths[i]);
        EXPECT_TRUE(utility::filesystem::FileExists(dataset.GetPaths(i)));
    }

    EXPECT_EQ(dataset.GetPrefix(), prefix);
    EXPECT_EQ(dataset.GetDataRoot(), data_root);
    EXPECT_EQ(dataset.GetDownloadDir(), download_dir);
    EXPECT_EQ(dataset.GetExtractDir(), extract_dir);
}

TEST(Dataset, DISABLED_RedwoodOfficePointClouds) {
    const std::string prefix = "OfficePointClouds";
    const std::string data_root =
            utility::filesystem::GetHomeDirectory() + "/open3d_data";
    const std::string download_dir = data_root + "/download/" + prefix;
    const std::string extract_dir = data_root + "/extract/" + prefix;

    data::OfficePointClouds dataset;
    EXPECT_TRUE(utility::filesystem::DirectoryExists(download_dir));

    std::vector<std::string> paths;
    paths.reserve(53);
    for (int i = 0; i < 53; ++i) {
        paths.push_back(extract_dir + "/cloud_bin_" + std::to_string(i) +
                        ".ply");
    }
    EXPECT_EQ(dataset.GetPaths(), paths);
    for (size_t i = 0; i < paths.size(); ++i) {
        EXPECT_EQ(dataset.GetPaths(i), paths[i]);
        EXPECT_TRUE(utility::filesystem::FileExists(dataset.GetPaths(i)));
    }

    EXPECT_EQ(dataset.GetPaths(), paths);
    for (size_t i = 0; i < paths.size(); ++i) {
        EXPECT_EQ(dataset.GetPaths(i), paths[i]);
        EXPECT_TRUE(utility::filesystem::FileExists(dataset.GetPaths(i)));
    }

    EXPECT_EQ(dataset.GetPrefix(), prefix);
    EXPECT_EQ(dataset.GetDataRoot(), data_root);
    EXPECT_EQ(dataset.GetDownloadDir(), download_dir);
    EXPECT_EQ(dataset.GetExtractDir(), extract_dir);
}

}  // namespace tests
}  // namespace open3d
