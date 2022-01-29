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

#include <string>

#include "open3d/utility/Download.h"
#include "open3d/utility/Extract.h"
#include "open3d/utility/FileSystem.h"
#include "open3d/utility/Logging.h"

namespace open3d {
namespace data {

std::string LocateDataRoot() {
    std::string data_root = "";
    if (const char* env_p = std::getenv("OPEN3D_DATA_ROOT")) {
        data_root = std::string(env_p);
    }
    if (data_root.empty()) {
        data_root = utility::filesystem::GetHomeDirectory() + "/open3d_data";
    }
    return data_root;
}

Dataset::Dataset(const std::string& prefix, const std::string& data_root)
    : prefix_(prefix) {
    if (data_root.empty()) {
        data_root_ = LocateDataRoot();
    } else {
        data_root_ = data_root;
    }
    if (prefix_.empty()) {
        utility::LogError("prefix cannot be empty.");
    }
}

SimpleDataset::SimpleDataset(const std::string& prefix,
                             const std::vector<std::string>& urls,
                             const std::string& md5,
                             const bool no_extract,
                             const std::string& data_root)
    : Dataset(prefix, data_root) {
    const std::string filename =
            utility::filesystem::GetFileNameWithoutDirectory(urls[0]);

    const bool is_extract_present =
            utility::filesystem::DirectoryExists(Dataset::GetExtractDir());

    if (!is_extract_present) {
        // `download_dir` is relative path from `${data_root}`.
        const std::string download_dir = "download/" + GetPrefix();
        const std::string download_file_path =
                utility::DownloadFromURL(urls, md5, download_dir, data_root_);

        // Extract / Copy data.
        if (!no_extract) {
            utility::Extract(download_file_path, Dataset::GetExtractDir());
        } else {
            utility::filesystem::MakeDirectoryHierarchy(
                    Dataset::GetExtractDir());
            utility::filesystem::CopyFile(download_file_path,
                                          Dataset::GetExtractDir());
        }
    }
}

DemoICPPointClouds::DemoICPPointClouds(const std::string& prefix,
                                       const std::string& data_root)
    : SimpleDataset(
              prefix,
              {"https://github.com/isl-org/open3d_downloads/releases/download/"
               "290122-demo-icp-pointclouds/DemoICPPointClouds.zip"},
              "76cf67ab1af942e3c4d5e97b9c2ae58f") {
    for (int i = 0; i < 3; ++i) {
        paths_.push_back(Dataset::GetExtractDir() + "/cloud_bin_" +
                         std::to_string(i) + ".pcd");
    }
}

std::string DemoICPPointClouds::GetPaths(size_t index) const {
    if (index > 2) {
        utility::LogError(
                "Invalid index. Expected index between 0 to 2 but got {}.",
                index);
    }
    return paths_[index];
}

DemoColoredICPPointClouds::DemoColoredICPPointClouds(
        const std::string& prefix, const std::string& data_root)
    : SimpleDataset(
              prefix,
              {"https://github.com/isl-org/open3d_downloads/releases/download/"
               "290122-demo-icp-pointclouds/DemoColoredICPPointClouds.zip"},
              "bf8d469e892d76f2e69e1213207c0e30") {
    paths_.push_back(Dataset::GetExtractDir() + "/frag_115.ply");
    paths_.push_back(Dataset::GetExtractDir() + "/frag_116.ply");
}

std::string DemoColoredICPPointClouds::GetPaths(size_t index) const {
    if (index > 1) {
        utility::LogError(
                "Invalid index. Expected index between 0 to 1 but got {}.",
                index);
    }
    return paths_[index];
}

DemoCropPointCloud::DemoCropPointCloud(const std::string& prefix,
                                       const std::string& data_root)
    : SimpleDataset(
              prefix,
              {"https://github.com/isl-org/open3d_downloads/releases/download/"
               "290122-demo-crop-pointcloud/DemoCropPointCloud.zip"},
              "12dbcdddd3f0865d8312929506135e23") {
    const std::string extract_dir = Dataset::GetExtractDir();
    path_pointcloud_ = extract_dir + "/fragment.ply";
    path_cropped_json_ = extract_dir + "/cropped.json";
}

DemoPointCloudFeatureMatching::DemoPointCloudFeatureMatching(
        const std::string& prefix, const std::string& data_root)
    : SimpleDataset(prefix,
                    {"https://github.com/isl-org/open3d_downloads/releases/"
                     "download/290122-demo-pointcloud-feature-matching/"
                     "DemoPointCloudFeatureMatching.zip"},
                    "02f0703ce0cbf4df78ce2602ae33fc79") {
    const std::string extract_dir = Dataset::GetExtractDir();
    paths_pointclouds_ = {extract_dir + "/cloud_bin_0.pcd",
                          extract_dir + "/cloud_bin_1.pcd"};
    paths_fpfh_features_ = {extract_dir + "/cloud_bin_0.fpfh.bin",
                            extract_dir + "/cloud_bin_1.fpfh.bin"};
    paths_l32d_features_ = {extract_dir + "/cloud_bin_0.d32.bin",
                            extract_dir + "/cloud_bin_1.d32.bin"};
}

DemoPoseGraphOptimization::DemoPoseGraphOptimization(
        const std::string& prefix, const std::string& data_root)
    : SimpleDataset(prefix,
                    {"https://github.com/isl-org/open3d_downloads/releases/"
                     "download/290122-demo-pose-graph-optimization/"
                     "DemoPoseGraphOptimization.zip"},
                    "af085b28d79dea7f0a50aef50c96b62c") {
    const std::string extract_dir = Dataset::GetExtractDir();
    path_pose_graph_fragment_ =
            extract_dir + "/pose_graph_example_fragment.json";
    path_pose_graph_global_ = extract_dir + "/pose_graph_example_global.json";
}

Armadillo::Armadillo(const std::string& prefix, const std::string& data_root)
    : SimpleDataset(prefix,
                    {"https://github.com/isl-org/open3d_downloads/releases/"
                     "download/stanford-mesh/Armadillo.ply"},
                    "9e68ff1b1cc914ed88cd84f6a8235021",
                    /*no_extract =*/true) {
    path_ = Dataset::GetExtractDir() + "/Armadillo.ply";
}

Bunny::Bunny(const std::string& prefix, const std::string& data_root)
    : SimpleDataset(prefix,
                    {"https://github.com/isl-org/open3d_downloads/releases/"
                     "download/stanford-mesh/Bunny.ply"},
                    "568f871d1a221ba6627569f1e6f9a3f2",
                    /*no_extract =*/true) {
    path_ = Dataset::GetExtractDir() + "/Bunny.ply";
}

RedwoodLivingRoomPointClouds::RedwoodLivingRoomPointClouds(
        const std::string& prefix, const std::string& data_root)
    : SimpleDataset(prefix,
                    {"http://redwood-data.org/indoor/data/"
                     "livingroom1-fragments-ply.zip",
                     "https://github.com/isl-org/open3d_downloads/releases/"
                     "download/redwood/livingroom1-fragments-ply.zip"},
                    "36e0eb23a66ccad6af52c05f8390d33e") {
    paths_.reserve(57);
    for (int i = 0; i < 57; ++i) {
        paths_.push_back(Dataset::GetExtractDir() + "/cloud_bin_" +
                         std::to_string(i) + ".ply");
    }
}

std::string RedwoodLivingRoomPointClouds::GetPaths(size_t index) const {
    if (index > 56) {
        utility::LogError(
                "Invalid index. Expected index between 0 to 56 but got {}.",
                index);
    }
    return paths_[index];
}

RedwoodOfficePointClouds::RedwoodOfficePointClouds(const std::string& prefix,
                                                   const std::string& data_root)
    : SimpleDataset(prefix,
                    {"http://redwood-data.org/indoor/data/"
                     "office1-fragments-ply.zip",
                     "https://github.com/isl-org/open3d_downloads/releases/"
                     "download/redwood/office1-fragments-ply.zip"},
                    "c519fe0495b3c731ebe38ae3a227ac25") {
    paths_.reserve(53);
    for (int i = 0; i < 53; ++i) {
        paths_.push_back(Dataset::GetExtractDir() + "/cloud_bin_" +
                         std::to_string(i) + ".ply");
    }
}

std::string RedwoodOfficePointClouds::GetPaths(size_t index) const {
    if (index > 52) {
        utility::LogError(
                "Invalid index. Expected index between 0 to 52 but got {}.",
                index);
    }
    return paths_[index];
}

}  // namespace data
}  // namespace open3d
