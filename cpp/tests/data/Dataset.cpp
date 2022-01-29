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
            "https://github.com/isl-org/open3d_downloads/releases/download/"
            "stanford-mesh/Bunny.ply"};
    const std::string md5 = "568f871d1a221ba6627569f1e6f9a3f2";

    data::SimpleDataset simple_dataset(prefix, url_mirrors, md5,
                                       /*no_extact*/ true);

    // Check if file is downloaded and extracted / copied.
    EXPECT_TRUE(utility::filesystem::FileExists(download_dir + "/Bunny.ply"));
    EXPECT_TRUE(utility::filesystem::FileExists(extract_dir + "/Bunny.ply"));

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
    EXPECT_EQ(demo_crop_pcd.GetPathCroppedJSON(),
              extract_dir + "/cropped.json");

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
    EXPECT_EQ(armadillo.GetPath(), extract_dir + "/Armadillo.ply");
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
    EXPECT_EQ(bunny.GetPath(), extract_dir + "/Bunny.ply");
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

TEST(Dataset, RedwoodLivingRoomPointClouds) {
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

TEST(Dataset, RedwoodOfficePointClouds) {
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
