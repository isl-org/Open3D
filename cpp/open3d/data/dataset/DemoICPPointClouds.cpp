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
        Open3DDownloadsPrefix() + "20220301-data/DemoICPPointClouds.zip",
        "596cffe5f9c587045e7397ad70754de9"};

DemoICPPointClouds::DemoICPPointClouds(const std::string& data_root)
    : DownloadDataset("DemoICPPointClouds", data_descriptor, data_root) {
    for (int i = 0; i < 3; ++i) {
        paths_.push_back(GetExtractDir() + "/cloud_bin_" + std::to_string(i) +
                         ".pcd");
    }
    transformation_log_path_ = GetExtractDir() + "/init.log";
}

std::string DemoICPPointClouds::GetPaths(size_t index) const {
    if (index > 2) {
        utility::LogError(
                "Invalid index. Expected index between 0 to 2 but got {}.",
                index);
    }
    return paths_[index];
}

}  // namespace data
}  // namespace open3d
