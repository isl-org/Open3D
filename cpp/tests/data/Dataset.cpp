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
    // Prefix cannot be empty.
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

TEST(Dataset, SimpleDataset) {
    const std::string prefix = "O3DTestSimpleDataset";
    const std::string data_root =
            utility::filesystem::GetHomeDirectory() + "/open3d_data";
    const std::string download_dir = data_root + "/download/" + prefix;
    const std::string extract_dir = data_root + "/extract/" + prefix;

    // Delete if files already exists.
    utility::filesystem::DeleteDirectory(data_root + "/download/" + prefix);
    utility::filesystem::DeleteDirectory(data_root + "/extract/" + prefix);

    const std::vector<std::string> url_mirrors = {
            "https://github.com/isl-org/open3d_downloads/releases/"
            "download/20220130-sample-meshs/BunnyMesh.ply"};
    const std::string md5 = "568f871d1a221ba6627569f1e6f9a3f2";

    data::SimpleDataset simple_dataset(prefix, url_mirrors, md5,
                                       /*no_extact*/ true);

    // Check if file is downloaded and extracted / copied.
    EXPECT_TRUE(
            utility::filesystem::FileExists(download_dir + "/BunnyMesh.ply"));
    EXPECT_TRUE(
            utility::filesystem::FileExists(extract_dir + "/BunnyMesh.ply"));

    // Basic methods.
    EXPECT_EQ(simple_dataset.GetPrefix(), prefix);
    EXPECT_EQ(simple_dataset.GetDataRoot(), data_root);
    EXPECT_EQ(simple_dataset.GetDownloadDir(), download_dir);
    EXPECT_EQ(simple_dataset.GetExtractDir(), extract_dir);

    // Delete dataset.
    utility::filesystem::DeleteDirectory(download_dir);
    utility::filesystem::DeleteDirectory(extract_dir);
}

TEST(Dataset, DemoICPPointClouds) {
    const std::string prefix = "O3DTestDemoICPPointClouds";
    const std::string data_root =
            utility::filesystem::GetHomeDirectory() + "/open3d_data";
    const std::string download_dir = data_root + "/download/" + prefix;
    const std::string extract_dir = data_root + "/extract/" + prefix;

    // Delete if files already exists.
    utility::filesystem::DeleteDirectory(data_root + "/download/" + prefix);
    utility::filesystem::DeleteDirectory(data_root + "/extract/" + prefix);

    data::DemoICPPointClouds demo_icp(prefix);
    // Check if downloaded.
    EXPECT_TRUE(utility::filesystem::DirectoryExists(download_dir));

    // Methods to get path.
    const std::vector<std::string> paths = {extract_dir + "/cloud_bin_0.pcd",
                                            extract_dir + "/cloud_bin_1.pcd",
                                            extract_dir + "/cloud_bin_2.pcd"};
    EXPECT_EQ(demo_icp.GetPaths(), paths);
    for (size_t i = 0; i < paths.size(); ++i) {
        EXPECT_EQ(demo_icp.GetPaths(i), paths[i]);
        // Check if the file actually exists.
        EXPECT_TRUE(utility::filesystem::FileExists(demo_icp.GetPaths(i)));
    }

    // Basic methods.
    EXPECT_EQ(demo_icp.GetPrefix(), prefix);
    EXPECT_EQ(demo_icp.GetDataRoot(), data_root);
    EXPECT_EQ(demo_icp.GetDownloadDir(), download_dir);
    EXPECT_EQ(demo_icp.GetExtractDir(), extract_dir);

    // Delete dataset.
    utility::filesystem::DeleteDirectory(download_dir);
    utility::filesystem::DeleteDirectory(extract_dir);
}

TEST(Dataset, DemoColoredICPPointClouds) {
    const std::string prefix = "O3DTestDemoColoredICPPointClouds";
    const std::string data_root =
            utility::filesystem::GetHomeDirectory() + "/open3d_data";
    const std::string download_dir = data_root + "/download/" + prefix;
    const std::string extract_dir = data_root + "/extract/" + prefix;

    // Delete if files already exists.
    utility::filesystem::DeleteDirectory(data_root + "/download/" + prefix);
    utility::filesystem::DeleteDirectory(data_root + "/extract/" + prefix);

    data::DemoColoredICPPointClouds demo_cicp(prefix);
    // Check if downloaded.
    EXPECT_TRUE(utility::filesystem::DirectoryExists(download_dir));

    // Methods to get path.
    const std::vector<std::string> paths = {extract_dir + "/frag_115.ply",
                                            extract_dir + "/frag_116.ply"};
    EXPECT_EQ(demo_cicp.GetPaths(), paths);
    for (size_t i = 0; i < paths.size(); ++i) {
        EXPECT_EQ(demo_cicp.GetPaths(i), paths[i]);
        // Check if the file actually exists.
        EXPECT_TRUE(utility::filesystem::FileExists(demo_cicp.GetPaths(i)));
    }

    // Basic methods.
    EXPECT_EQ(demo_cicp.GetPrefix(), prefix);
    EXPECT_EQ(demo_cicp.GetDataRoot(), data_root);
    EXPECT_EQ(demo_cicp.GetDownloadDir(), download_dir);
    EXPECT_EQ(demo_cicp.GetExtractDir(), extract_dir);

    // Delete dataset.
    utility::filesystem::DeleteDirectory(download_dir);
    utility::filesystem::DeleteDirectory(extract_dir);
}

TEST(Dataset, DemoCropPointCloud) {
    const std::string prefix = "O3DTestDemoCropPointCloud";
    const std::string data_root =
            utility::filesystem::GetHomeDirectory() + "/open3d_data";
    const std::string download_dir = data_root + "/download/" + prefix;
    const std::string extract_dir = data_root + "/extract/" + prefix;

    // Delete if files already exists.
    utility::filesystem::DeleteDirectory(data_root + "/download/" + prefix);
    utility::filesystem::DeleteDirectory(data_root + "/extract/" + prefix);

    data::DemoCropPointCloud demo_crop_pcd(prefix);
    // Check if downloaded.
    EXPECT_TRUE(utility::filesystem::DirectoryExists(download_dir));

    // Methods to get path.
    EXPECT_EQ(demo_crop_pcd.GetPathPointCloud(), extract_dir + "/fragment.ply");
    EXPECT_TRUE(
            utility::filesystem::FileExists(demo_crop_pcd.GetPathPointCloud()));
    EXPECT_EQ(demo_crop_pcd.GetPathCroppedJSON(),
              extract_dir + "/cropped.json");
    EXPECT_TRUE(utility::filesystem::FileExists(
            demo_crop_pcd.GetPathCroppedJSON()));

    // Basic methods.
    EXPECT_EQ(demo_crop_pcd.GetPrefix(), prefix);
    EXPECT_EQ(demo_crop_pcd.GetDataRoot(), data_root);
    EXPECT_EQ(demo_crop_pcd.GetDownloadDir(), download_dir);
    EXPECT_EQ(demo_crop_pcd.GetExtractDir(), extract_dir);

    // Delete dataset.
    utility::filesystem::DeleteDirectory(download_dir);
    utility::filesystem::DeleteDirectory(extract_dir);
}

TEST(Dataset, DemoPointCloudFeatureMatching) {
    const std::string prefix = "O3DTestDemoPointCloudFeatureMatching";
    const std::string data_root =
            utility::filesystem::GetHomeDirectory() + "/open3d_data";
    const std::string download_dir = data_root + "/download/" + prefix;
    const std::string extract_dir = data_root + "/extract/" + prefix;

    // Delete if files already exists.
    utility::filesystem::DeleteDirectory(data_root + "/download/" + prefix);
    utility::filesystem::DeleteDirectory(data_root + "/extract/" + prefix);

    data::DemoPointCloudFeatureMatching demo_feature_matching(prefix);
    // Check if downloaded.
    EXPECT_TRUE(utility::filesystem::DirectoryExists(download_dir));

    // Methods to get path.
    const std::vector<std::string> paths_pointclouds = {
            extract_dir + "/cloud_bin_0.pcd", extract_dir + "/cloud_bin_1.pcd"};
    EXPECT_EQ(demo_feature_matching.GetPathsPointClouds(), paths_pointclouds);

    const std::vector<std::string> paths_fpfh_features = {
            extract_dir + "/cloud_bin_0.fpfh.bin",
            extract_dir + "/cloud_bin_1.fpfh.bin"};
    EXPECT_EQ(demo_feature_matching.GetPathsFPFHFeatures(),
              paths_fpfh_features);

    const std::vector<std::string> paths_l32d_features = {
            extract_dir + "/cloud_bin_0.d32.bin",
            extract_dir + "/cloud_bin_1.d32.bin"};
    EXPECT_EQ(demo_feature_matching.GetPathsL32DFeatures(),
              paths_l32d_features);

    // Basic methods.
    EXPECT_EQ(demo_feature_matching.GetPrefix(), prefix);
    EXPECT_EQ(demo_feature_matching.GetDataRoot(), data_root);
    EXPECT_EQ(demo_feature_matching.GetDownloadDir(), download_dir);
    EXPECT_EQ(demo_feature_matching.GetExtractDir(), extract_dir);

    // Delete dataset.
    utility::filesystem::DeleteDirectory(download_dir);
    utility::filesystem::DeleteDirectory(extract_dir);
}

TEST(Dataset, DemoPoseGraphOptimization) {
    const std::string prefix = "O3DTestDemoPoseGraphOptimization";
    const std::string data_root =
            utility::filesystem::GetHomeDirectory() + "/open3d_data";
    const std::string download_dir = data_root + "/download/" + prefix;
    const std::string extract_dir = data_root + "/extract/" + prefix;

    // Delete if files already exists.
    utility::filesystem::DeleteDirectory(data_root + "/download/" + prefix);
    utility::filesystem::DeleteDirectory(data_root + "/extract/" + prefix);

    data::DemoPoseGraphOptimization demo_pose_optimization(prefix);
    // Check if downloaded.
    EXPECT_TRUE(utility::filesystem::DirectoryExists(download_dir));

    // Methods to get path.
    EXPECT_EQ(demo_pose_optimization.GetPathPoseGraphFragment(),
              extract_dir + "/pose_graph_example_fragment.json");
    EXPECT_EQ(demo_pose_optimization.GetPathPoseGraphGlobal(),
              extract_dir + "/pose_graph_example_global.json");

    // Basic methods.
    EXPECT_EQ(demo_pose_optimization.GetPrefix(), prefix);
    EXPECT_EQ(demo_pose_optimization.GetDataRoot(), data_root);
    EXPECT_EQ(demo_pose_optimization.GetDownloadDir(), download_dir);
    EXPECT_EQ(demo_pose_optimization.GetExtractDir(), extract_dir);

    // Delete dataset.
    utility::filesystem::DeleteDirectory(download_dir);
    utility::filesystem::DeleteDirectory(extract_dir);
}

TEST(Dataset, SamplePointCloudPCD) {
    const std::string prefix = "O3DTestSamplePointCloudPCD";
    const std::string data_root =
            utility::filesystem::GetHomeDirectory() + "/open3d_data";
    const std::string download_dir = data_root + "/download/" + prefix;
    const std::string extract_dir = data_root + "/extract/" + prefix;

    // Delete if files already exists.
    utility::filesystem::DeleteDirectory(data_root + "/download/" + prefix);
    utility::filesystem::DeleteDirectory(data_root + "/extract/" + prefix);

    data::SamplePointCloudPCD pointcloud_pcd(prefix);
    // Check if downloaded.
    EXPECT_TRUE(utility::filesystem::DirectoryExists(download_dir));

    // Method to get path.
    EXPECT_EQ(pointcloud_pcd.GetPath(), extract_dir + "/fragment.pcd");
    // Check if the file actually exists.
    EXPECT_TRUE(utility::filesystem::FileExists(pointcloud_pcd.GetPath()));

    // Basic method.
    EXPECT_EQ(pointcloud_pcd.GetPrefix(), prefix);
    EXPECT_EQ(pointcloud_pcd.GetDataRoot(), data_root);
    EXPECT_EQ(pointcloud_pcd.GetDownloadDir(), download_dir);
    EXPECT_EQ(pointcloud_pcd.GetExtractDir(), extract_dir);

    // Delete dataset.
    utility::filesystem::DeleteDirectory(download_dir);
    utility::filesystem::DeleteDirectory(extract_dir);
}

TEST(Dataset, SamplePointCloudPLY) {
    const std::string prefix = "O3DTestSamplePointCloudPLY";
    const std::string data_root =
            utility::filesystem::GetHomeDirectory() + "/open3d_data";
    const std::string download_dir = data_root + "/download/" + prefix;
    const std::string extract_dir = data_root + "/extract/" + prefix;

    // Delete if files already exists.
    utility::filesystem::DeleteDirectory(data_root + "/download/" + prefix);
    utility::filesystem::DeleteDirectory(data_root + "/extract/" + prefix);

    data::SamplePointCloudPLY pointcloud_ply(prefix);
    // Check if downloaded.
    EXPECT_TRUE(utility::filesystem::DirectoryExists(download_dir));

    // Method to get path.
    EXPECT_EQ(pointcloud_ply.GetPath(), extract_dir + "/fragment.ply");
    // Check if the file actually exists.
    EXPECT_TRUE(utility::filesystem::FileExists(pointcloud_ply.GetPath()));

    // Basic method.
    EXPECT_EQ(pointcloud_ply.GetPrefix(), prefix);
    EXPECT_EQ(pointcloud_ply.GetDataRoot(), data_root);
    EXPECT_EQ(pointcloud_ply.GetDownloadDir(), download_dir);
    EXPECT_EQ(pointcloud_ply.GetExtractDir(), extract_dir);

    // Delete dataset.
    utility::filesystem::DeleteDirectory(download_dir);
    utility::filesystem::DeleteDirectory(extract_dir);
}

TEST(Dataset, SampleRGBDImageNYU) {
    const std::string prefix = "O3DTestSampleRGBDImageNYU";
    const std::string data_root =
            utility::filesystem::GetHomeDirectory() + "/open3d_data";
    const std::string download_dir = data_root + "/download/" + prefix;
    const std::string extract_dir = data_root + "/extract/" + prefix;

    // Delete if files already exists.
    utility::filesystem::DeleteDirectory(data_root + "/download/" + prefix);
    utility::filesystem::DeleteDirectory(data_root + "/extract/" + prefix);

    data::SampleRGBDImageNYU rgbd_nyu(prefix);
    // Check if downloaded.
    EXPECT_TRUE(utility::filesystem::DirectoryExists(download_dir));

    // Method to get path.
    EXPECT_EQ(rgbd_nyu.GetPathColor(), extract_dir + "/NYU_color.ppm");
    EXPECT_TRUE(utility::filesystem::FileExists(rgbd_nyu.GetPathColor()));
    EXPECT_EQ(rgbd_nyu.GetPathDepth(), extract_dir + "/NYU_depth.pgm");
    EXPECT_TRUE(utility::filesystem::FileExists(rgbd_nyu.GetPathDepth()));

    // Basic method.
    EXPECT_EQ(rgbd_nyu.GetPrefix(), prefix);
    EXPECT_EQ(rgbd_nyu.GetDataRoot(), data_root);
    EXPECT_EQ(rgbd_nyu.GetDownloadDir(), download_dir);
    EXPECT_EQ(rgbd_nyu.GetExtractDir(), extract_dir);

    // Delete dataset.
    utility::filesystem::DeleteDirectory(download_dir);
    utility::filesystem::DeleteDirectory(extract_dir);
}

TEST(Dataset, SampleRGBDImageSUN) {
    const std::string prefix = "O3DTestSampleRGBDImageSUN";
    const std::string data_root =
            utility::filesystem::GetHomeDirectory() + "/open3d_data";
    const std::string download_dir = data_root + "/download/" + prefix;
    const std::string extract_dir = data_root + "/extract/" + prefix;

    // Delete if files already exists.
    utility::filesystem::DeleteDirectory(data_root + "/download/" + prefix);
    utility::filesystem::DeleteDirectory(data_root + "/extract/" + prefix);

    data::SampleRGBDImageSUN rgbd_sun(prefix);
    // Check if downloaded.
    EXPECT_TRUE(utility::filesystem::DirectoryExists(download_dir));

    // Method to get path.
    EXPECT_EQ(rgbd_sun.GetPathColor(), extract_dir + "/SUN_color.jpg");
    EXPECT_TRUE(utility::filesystem::FileExists(rgbd_sun.GetPathColor()));
    EXPECT_EQ(rgbd_sun.GetPathDepth(), extract_dir + "/SUN_depth.png");
    EXPECT_TRUE(utility::filesystem::FileExists(rgbd_sun.GetPathDepth()));

    // Basic method.
    EXPECT_EQ(rgbd_sun.GetPrefix(), prefix);
    EXPECT_EQ(rgbd_sun.GetDataRoot(), data_root);
    EXPECT_EQ(rgbd_sun.GetDownloadDir(), download_dir);
    EXPECT_EQ(rgbd_sun.GetExtractDir(), extract_dir);

    // Delete dataset.
    utility::filesystem::DeleteDirectory(download_dir);
    utility::filesystem::DeleteDirectory(extract_dir);
}

TEST(Dataset, SampleRGBDImageTUM) {
    const std::string prefix = "O3DTestSampleRGBDImageTUM";
    const std::string data_root =
            utility::filesystem::GetHomeDirectory() + "/open3d_data";
    const std::string download_dir = data_root + "/download/" + prefix;
    const std::string extract_dir = data_root + "/extract/" + prefix;

    // Delete if files already exists.
    utility::filesystem::DeleteDirectory(data_root + "/download/" + prefix);
    utility::filesystem::DeleteDirectory(data_root + "/extract/" + prefix);

    data::SampleRGBDImageTUM rgbd_tum(prefix);
    // Check if downloaded.
    EXPECT_TRUE(utility::filesystem::DirectoryExists(download_dir));

    // Method to get path.
    EXPECT_EQ(rgbd_tum.GetPathColor(), extract_dir + "/TUM_color.png");
    EXPECT_TRUE(utility::filesystem::FileExists(rgbd_tum.GetPathColor()));
    EXPECT_EQ(rgbd_tum.GetPathDepth(), extract_dir + "/TUM_depth.png");
    EXPECT_TRUE(utility::filesystem::FileExists(rgbd_tum.GetPathDepth()));

    // Basic method.
    EXPECT_EQ(rgbd_tum.GetPrefix(), prefix);
    EXPECT_EQ(rgbd_tum.GetDataRoot(), data_root);
    EXPECT_EQ(rgbd_tum.GetDownloadDir(), download_dir);
    EXPECT_EQ(rgbd_tum.GetExtractDir(), extract_dir);

    // Delete dataset.
    utility::filesystem::DeleteDirectory(download_dir);
    utility::filesystem::DeleteDirectory(extract_dir);
}

TEST(Dataset, SampleRGBDDatasetICL) {
    const std::string prefix = "O3DTestSampleRGBDDatasetICL";
    const std::string data_root =
            utility::filesystem::GetHomeDirectory() + "/open3d_data";
    const std::string download_dir = data_root + "/download/" + prefix;
    const std::string extract_dir = data_root + "/extract/" + prefix;

    // Delete if files already exists.
    utility::filesystem::DeleteDirectory(data_root + "/download/" + prefix);
    utility::filesystem::DeleteDirectory(data_root + "/extract/" + prefix);

    data::SampleRGBDDatasetICL rgbd_icl(prefix);
    // Check if downloaded.
    EXPECT_TRUE(utility::filesystem::DirectoryExists(download_dir));

    // Method to get path.
    const std::vector<std::string> paths_color = {
            extract_dir + "/color/00000.jpg", extract_dir + "/color/00001.jpg",
            extract_dir + "/color/00002.jpg", extract_dir + "/color/00003.jpg",
            extract_dir + "/color/00004.jpg"};
    EXPECT_EQ(rgbd_icl.GetPathsColor(), paths_color);

    const std::vector<std::string> paths_depth = {
            extract_dir + "/depth/00000.png", extract_dir + "/depth/00001.png",
            extract_dir + "/depth/00002.png", extract_dir + "/depth/00003.png",
            extract_dir + "/depth/00004.png"};
    EXPECT_EQ(rgbd_icl.GetPathsDepth(), paths_depth);
    for (size_t i = 0; i < paths_color.size(); ++i) {
        EXPECT_TRUE(
                utility::filesystem::FileExists(rgbd_icl.GetPathsColor()[i]));
        EXPECT_TRUE(
                utility::filesystem::FileExists(rgbd_icl.GetPathsDepth()[i]));
    }

    EXPECT_EQ(rgbd_icl.GetPathTrajectoryLog(), extract_dir + "/trajectory.log");
    EXPECT_TRUE(
            utility::filesystem::FileExists(rgbd_icl.GetPathTrajectoryLog()));

    EXPECT_EQ(rgbd_icl.GetPathOdometryLog(), extract_dir + "/odometry.log");
    EXPECT_TRUE(utility::filesystem::FileExists(rgbd_icl.GetPathOdometryLog()));

    EXPECT_EQ(rgbd_icl.GetPathRGBDMatch(), extract_dir + "/rgbd.match");
    EXPECT_TRUE(utility::filesystem::FileExists(rgbd_icl.GetPathRGBDMatch()));

    EXPECT_EQ(rgbd_icl.GetPathReconstruction(),
              extract_dir + "/example_tsdf_pcd.ply");
    EXPECT_TRUE(
            utility::filesystem::FileExists(rgbd_icl.GetPathReconstruction()));

    // Basic method.
    EXPECT_EQ(rgbd_icl.GetPrefix(), prefix);
    EXPECT_EQ(rgbd_icl.GetDataRoot(), data_root);
    EXPECT_EQ(rgbd_icl.GetDownloadDir(), download_dir);
    EXPECT_EQ(rgbd_icl.GetExtractDir(), extract_dir);

    // Delete dataset.
    utility::filesystem::DeleteDirectory(download_dir);
    utility::filesystem::DeleteDirectory(extract_dir);
}

TEST(Dataset, SampleFountainRGBDDataset) {
    const std::string prefix = "O3DTestSampleFountainRGBDDataset";
    const std::string data_root =
            utility::filesystem::GetHomeDirectory() + "/open3d_data";
    const std::string download_dir = data_root + "/download/" + prefix;
    const std::string extract_dir = data_root + "/extract/" + prefix;

    // Delete if files already exists.
    utility::filesystem::DeleteDirectory(data_root + "/download/" + prefix);
    utility::filesystem::DeleteDirectory(data_root + "/extract/" + prefix);

    data::SampleFountainRGBDDataset rgbd_fountain(prefix);
    // Check if downloaded.
    EXPECT_TRUE(utility::filesystem::DirectoryExists(download_dir));

    // Method to get path.
    const std::vector<std::string> paths_color = {
            extract_dir + "/image/0000010-000001228920.jpg",
            extract_dir + "/image/0000368-000050112627.jpg",
            extract_dir + "/image/0000722-000098450147.jpg",
            extract_dir + "/image/0000031-000004096400.jpg",
            extract_dir + "/image/0000412-000056120680.jpg",
            extract_dir + "/image/0000771-000105140933.jpg",
            extract_dir + "/image/0000044-000005871507.jpg",
            extract_dir + "/image/0000429-000058441973.jpg",
            extract_dir + "/image/0000792-000108008413.jpg",
            extract_dir + "/image/0000064-000008602440.jpg",
            extract_dir + "/image/0000474-000064586573.jpg",
            extract_dir + "/image/0000818-000111558627.jpg",
            extract_dir + "/image/0000110-000014883587.jpg",
            extract_dir + "/image/0000487-000066361680.jpg",
            extract_dir + "/image/0000849-000115791573.jpg",
            extract_dir + "/image/0000156-000021164733.jpg",
            extract_dir + "/image/0000526-000071687000.jpg",
            extract_dir + "/image/0000883-000120434160.jpg",
            extract_dir + "/image/0000200-000027172787.jpg",
            extract_dir + "/image/0000549-000074827573.jpg",
            extract_dir + "/image/0000896-000122209267.jpg",
            extract_dir + "/image/0000215-000029220987.jpg",
            extract_dir + "/image/0000582-000079333613.jpg",
            extract_dir + "/image/0000935-000127534587.jpg",
            extract_dir + "/image/0000255-000034682853.jpg",
            extract_dir + "/image/0000630-000085887853.jpg",
            extract_dir + "/image/0000985-000134361920.jpg",
            extract_dir + "/image/0000299-000040690907.jpg",
            extract_dir + "/image/0000655-000089301520.jpg",
            extract_dir + "/image/0001028-000140233427.jpg",
            extract_dir + "/image/0000331-000045060400.jpg",
            extract_dir + "/image/0000703-000095855760.jpg",
            extract_dir + "/image/0001061-000144739467.jpg"};
    EXPECT_EQ(rgbd_fountain.GetPathsColor(), paths_color);

    const std::vector<std::string> paths_depth = {
            extract_dir + "/depth/0000038-000001234662.png",
            extract_dir + "/depth/0001503-000050120614.png",
            extract_dir + "/depth/0002951-000098439288.png",
            extract_dir + "/depth/0000124-000004104418.png",
            extract_dir + "/depth/0001683-000056127079.png",
            extract_dir + "/depth/0003152-000105146507.png",
            extract_dir + "/depth/0000177-000005872988.png",
            extract_dir + "/depth/0001752-000058429557.png",
            extract_dir + "/depth/0003238-000108016262.png",
            extract_dir + "/depth/0000259-000008609267.png",
            extract_dir + "/depth/0001937-000064602868.png",
            extract_dir + "/depth/0003344-000111553403.png",
            extract_dir + "/depth/0000447-000014882686.png",
            extract_dir + "/depth/0001990-000066371438.png",
            extract_dir + "/depth/0003471-000115791298.png",
            extract_dir + "/depth/0000635-000021156105.png",
            extract_dir + "/depth/0002149-000071677149.png",
            extract_dir + "/depth/0003610-000120429623.png",
            extract_dir + "/depth/0000815-000027162570.png",
            extract_dir + "/depth/0002243-000074813859.png",
            extract_dir + "/depth/0003663-000122198194.png",
            extract_dir + "/depth/0000877-000029231463.png",
            extract_dir + "/depth/0002378-000079318707.png",
            extract_dir + "/depth/0003823-000127537274.png",
            extract_dir + "/depth/0001040-000034670651.png",
            extract_dir + "/depth/0002575-000085892450.png",
            extract_dir + "/depth/0004028-000134377970.png",
            extract_dir + "/depth/0001220-000040677116.png",
            extract_dir + "/depth/0002677-000089296113.png",
            extract_dir + "/depth/0004203-000140217589.png",
            extract_dir + "/depth/0001351-000045048488.png",
            extract_dir + "/depth/0002874-000095869855.png",
            extract_dir + "/depth/0004339-000144755807.png"};
    EXPECT_EQ(rgbd_fountain.GetPathsDepth(), paths_depth);

    for (auto& path_color : rgbd_fountain.GetPathsColor()) {
        EXPECT_TRUE(utility::filesystem::FileExists(path_color));
    }
    for (auto& path_depth : rgbd_fountain.GetPathsDepth()) {
        EXPECT_TRUE(utility::filesystem::FileExists(path_depth));
    }

    EXPECT_EQ(rgbd_fountain.GetPathKeyframePosesLog(),
              extract_dir + "/scene/key.log");
    EXPECT_TRUE(utility::filesystem::FileExists(
            rgbd_fountain.GetPathKeyframePosesLog()));

    EXPECT_EQ(rgbd_fountain.GetPathReconstruction(),
              extract_dir + "/scene/integrated.ply");
    EXPECT_TRUE(utility::filesystem::FileExists(
            rgbd_fountain.GetPathReconstruction()));

    // Basic method.
    EXPECT_EQ(rgbd_fountain.GetPrefix(), prefix);
    EXPECT_EQ(rgbd_fountain.GetDataRoot(), data_root);
    EXPECT_EQ(rgbd_fountain.GetDownloadDir(), download_dir);
    EXPECT_EQ(rgbd_fountain.GetExtractDir(), extract_dir);

    // Delete dataset.
    utility::filesystem::DeleteDirectory(download_dir);
    utility::filesystem::DeleteDirectory(extract_dir);
}

TEST(Dataset, Eagle) {
    const std::string prefix = "O3DTestEagle";
    const std::string data_root =
            utility::filesystem::GetHomeDirectory() + "/open3d_data";
    const std::string download_dir = data_root + "/download/" + prefix;
    const std::string extract_dir = data_root + "/extract/" + prefix;

    // Delete if files already exists.
    utility::filesystem::DeleteDirectory(data_root + "/download/" + prefix);
    utility::filesystem::DeleteDirectory(data_root + "/extract/" + prefix);

    data::Eagle eagle(prefix);
    // Check if downloaded.
    EXPECT_TRUE(utility::filesystem::DirectoryExists(download_dir));

    // Method to get path.
    EXPECT_EQ(eagle.GetPath(), extract_dir + "/EaglePointCloud.ply");
    // Check if the file actually exists.
    EXPECT_TRUE(utility::filesystem::FileExists(eagle.GetPath()));

    // Basic method.
    EXPECT_EQ(eagle.GetPrefix(), prefix);
    EXPECT_EQ(eagle.GetDataRoot(), data_root);
    EXPECT_EQ(eagle.GetDownloadDir(), download_dir);
    EXPECT_EQ(eagle.GetExtractDir(), extract_dir);

    // Delete dataset.
    utility::filesystem::DeleteDirectory(download_dir);
    utility::filesystem::DeleteDirectory(extract_dir);
}

TEST(Dataset, Armadillo) {
    const std::string prefix = "O3DTestArmadillo";
    const std::string data_root =
            utility::filesystem::GetHomeDirectory() + "/open3d_data";
    const std::string download_dir = data_root + "/download/" + prefix;
    const std::string extract_dir = data_root + "/extract/" + prefix;

    // Delete if files already exists.
    utility::filesystem::DeleteDirectory(data_root + "/download/" + prefix);
    utility::filesystem::DeleteDirectory(data_root + "/extract/" + prefix);

    data::Armadillo armadillo(prefix);
    // Check if downloaded.
    EXPECT_TRUE(utility::filesystem::DirectoryExists(download_dir));

    // Method to get path.
    EXPECT_EQ(armadillo.GetPath(), extract_dir + "/ArmadilloMesh.ply");
    // Check if the file actually exists.
    EXPECT_TRUE(utility::filesystem::FileExists(armadillo.GetPath()));

    // Basic method.
    EXPECT_EQ(armadillo.GetPrefix(), prefix);
    EXPECT_EQ(armadillo.GetDataRoot(), data_root);
    EXPECT_EQ(armadillo.GetDownloadDir(), download_dir);
    EXPECT_EQ(armadillo.GetExtractDir(), extract_dir);

    // Delete dataset.
    utility::filesystem::DeleteDirectory(download_dir);
    utility::filesystem::DeleteDirectory(extract_dir);
}

TEST(Dataset, Bunny) {
    const std::string prefix = "O3DTestBunny";
    const std::string data_root =
            utility::filesystem::GetHomeDirectory() + "/open3d_data";
    const std::string download_dir = data_root + "/download/" + prefix;
    const std::string extract_dir = data_root + "/extract/" + prefix;

    // Delete if files already exists.
    utility::filesystem::DeleteDirectory(data_root + "/download/" + prefix);
    utility::filesystem::DeleteDirectory(data_root + "/extract/" + prefix);

    data::Bunny bunny(prefix);
    // Check if downloaded.
    EXPECT_TRUE(utility::filesystem::DirectoryExists(download_dir));

    // Method to get path.
    EXPECT_EQ(bunny.GetPath(), extract_dir + "/BunnyMesh.ply");
    // Check if the file actually exists.
    EXPECT_TRUE(utility::filesystem::FileExists(bunny.GetPath()));

    // Basic method.
    EXPECT_EQ(bunny.GetPrefix(), prefix);
    EXPECT_EQ(bunny.GetDataRoot(), data_root);
    EXPECT_EQ(bunny.GetDownloadDir(), download_dir);
    EXPECT_EQ(bunny.GetExtractDir(), extract_dir);

    // Delete dataset.
    utility::filesystem::DeleteDirectory(download_dir);
    utility::filesystem::DeleteDirectory(extract_dir);
}

TEST(Dataset, Knot) {
    const std::string prefix = "O3DTestKnot";
    const std::string data_root =
            utility::filesystem::GetHomeDirectory() + "/open3d_data";
    const std::string download_dir = data_root + "/download/" + prefix;
    const std::string extract_dir = data_root + "/extract/" + prefix;

    // Delete if files already exists.
    utility::filesystem::DeleteDirectory(data_root + "/download/" + prefix);
    utility::filesystem::DeleteDirectory(data_root + "/extract/" + prefix);

    data::Knot knot(prefix);
    // Check if downloaded.
    EXPECT_TRUE(utility::filesystem::DirectoryExists(download_dir));

    // Method to get path.
    EXPECT_EQ(knot.GetPath(), extract_dir + "/KnotMesh.ply");
    // Check if the file actually exists.
    EXPECT_TRUE(utility::filesystem::FileExists(knot.GetPath()));

    // Basic method.
    EXPECT_EQ(knot.GetPrefix(), prefix);
    EXPECT_EQ(knot.GetDataRoot(), data_root);
    EXPECT_EQ(knot.GetDownloadDir(), download_dir);
    EXPECT_EQ(knot.GetExtractDir(), extract_dir);

    // Delete dataset.
    utility::filesystem::DeleteDirectory(download_dir);
    utility::filesystem::DeleteDirectory(extract_dir);
}

TEST(Dataset, Juneau) {
    const std::string prefix = "O3DTestJuneau";
    const std::string data_root =
            utility::filesystem::GetHomeDirectory() + "/open3d_data";
    const std::string download_dir = data_root + "/download/" + prefix;
    const std::string extract_dir = data_root + "/extract/" + prefix;

    // Delete if files already exists.
    utility::filesystem::DeleteDirectory(data_root + "/download/" + prefix);
    utility::filesystem::DeleteDirectory(data_root + "/extract/" + prefix);

    data::Juneau juneau(prefix);
    // Check if downloaded.
    EXPECT_TRUE(utility::filesystem::DirectoryExists(download_dir));

    // Method to get path.
    EXPECT_EQ(juneau.GetPath(), extract_dir + "/JuneauImage.jpg");
    // Check if the file actually exists.
    EXPECT_TRUE(utility::filesystem::FileExists(juneau.GetPath()));

    // Basic method.
    EXPECT_EQ(juneau.GetPrefix(), prefix);
    EXPECT_EQ(juneau.GetDataRoot(), data_root);
    EXPECT_EQ(juneau.GetDownloadDir(), download_dir);
    EXPECT_EQ(juneau.GetExtractDir(), extract_dir);

    // Delete dataset.
    utility::filesystem::DeleteDirectory(download_dir);
    utility::filesystem::DeleteDirectory(extract_dir);
}

TEST(Dataset, DISABLED_RedwoodLivingRoomPointClouds) {
    const std::string prefix = "O3DTestRedwoodLivingRoomPointClouds";
    const std::string data_root =
            utility::filesystem::GetHomeDirectory() + "/open3d_data";
    const std::string download_dir = data_root + "/download/" + prefix;
    const std::string extract_dir = data_root + "/extract/" + prefix;

    // Delete if files already exists.
    utility::filesystem::DeleteDirectory(data_root + "/download/" + prefix);
    utility::filesystem::DeleteDirectory(data_root + "/extract/" + prefix);

    data::RedwoodLivingRoomPointClouds living_room(prefix);
    // Check if downloaded.
    EXPECT_TRUE(utility::filesystem::DirectoryExists(download_dir));

    // Methods to get path.
    std::vector<std::string> paths;
    paths.reserve(57);
    for (int i = 0; i < 57; ++i) {
        paths.push_back(extract_dir + "/cloud_bin_" + std::to_string(i) +
                        ".ply");
    }
    EXPECT_EQ(living_room.GetPaths(), paths);
    for (size_t i = 0; i < paths.size(); ++i) {
        EXPECT_EQ(living_room.GetPaths(i), paths[i]);
        // Check if the file actually exists.
        EXPECT_TRUE(utility::filesystem::FileExists(living_room.GetPaths(i)));
    }

    // Basic methods.
    EXPECT_EQ(living_room.GetPrefix(), prefix);
    EXPECT_EQ(living_room.GetDataRoot(), data_root);
    EXPECT_EQ(living_room.GetDownloadDir(), download_dir);
    EXPECT_EQ(living_room.GetExtractDir(), extract_dir);

    // Delete dataset.
    utility::filesystem::DeleteDirectory(download_dir);
    utility::filesystem::DeleteDirectory(extract_dir);
}

TEST(Dataset, DISABLED_RedwoodOfficePointClouds) {
    const std::string prefix = "O3DTestRedwoodOfficePointClouds";
    const std::string data_root =
            utility::filesystem::GetHomeDirectory() + "/open3d_data";
    const std::string download_dir = data_root + "/download/" + prefix;
    const std::string extract_dir = data_root + "/extract/" + prefix;

    // Delete if files already exists.
    utility::filesystem::DeleteDirectory(data_root + "/download/" + prefix);
    utility::filesystem::DeleteDirectory(data_root + "/extract/" + prefix);

    data::RedwoodOfficePointClouds office(prefix);
    // Check if downloaded.
    EXPECT_TRUE(utility::filesystem::DirectoryExists(download_dir));

    // Methods to get path.
    std::vector<std::string> paths;
    paths.reserve(53);
    for (int i = 0; i < 53; ++i) {
        paths.push_back(extract_dir + "/cloud_bin_" + std::to_string(i) +
                        ".ply");
    }
    EXPECT_EQ(office.GetPaths(), paths);
    for (size_t i = 0; i < paths.size(); ++i) {
        EXPECT_EQ(office.GetPaths(i), paths[i]);
        // Check if the file actually exists.
        EXPECT_TRUE(utility::filesystem::FileExists(office.GetPaths(i)));
    }

    EXPECT_EQ(office.GetPaths(), paths);
    for (size_t i = 0; i < paths.size(); ++i) {
        EXPECT_EQ(office.GetPaths(i), paths[i]);
        // Check if the file actually exists.
        EXPECT_TRUE(utility::filesystem::FileExists(office.GetPaths(i)));
    }

    // Basic methods.
    EXPECT_EQ(office.GetPrefix(), prefix);
    EXPECT_EQ(office.GetDataRoot(), data_root);
    EXPECT_EQ(office.GetDownloadDir(), download_dir);
    EXPECT_EQ(office.GetExtractDir(), extract_dir);

    // Delete dataset.
    utility::filesystem::DeleteDirectory(download_dir);
    utility::filesystem::DeleteDirectory(extract_dir);
}

}  // namespace tests
}  // namespace open3d
