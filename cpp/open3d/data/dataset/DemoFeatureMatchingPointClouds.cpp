// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <string>
#include <vector>

#include "open3d/data/Dataset.h"
#include "open3d/utility/Logging.h"

namespace open3d {
namespace data {

const static DataDescriptor data_descriptor = {
        Open3DDownloadsPrefix() +
                "20220201-data/DemoFeatureMatchingPointClouds.zip",
        "02f0703ce0cbf4df78ce2602ae33fc79"};

DemoFeatureMatchingPointClouds::DemoFeatureMatchingPointClouds(
        const std::string& data_root)
    : DownloadDataset(
              "DemoFeatureMatchingPointClouds", data_descriptor, data_root) {
    const std::string extract_dir = GetExtractDir();
    point_cloud_paths_ = {extract_dir + "/cloud_bin_0.pcd",
                          extract_dir + "/cloud_bin_1.pcd"};
    fpfh_feature_paths_ = {extract_dir + "/cloud_bin_0.fpfh.bin",
                           extract_dir + "/cloud_bin_1.fpfh.bin"};
    l32d_feature_paths_ = {extract_dir + "/cloud_bin_0.d32.bin",
                           extract_dir + "/cloud_bin_1.d32.bin"};
}

}  // namespace data
}  // namespace open3d
