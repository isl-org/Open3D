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
        Open3DDownloadsPrefix() + "20220201-data/DemoColoredICPPointClouds.zip",
        "bf8d469e892d76f2e69e1213207c0e30"};

DemoColoredICPPointClouds::DemoColoredICPPointClouds(
        const std::string& data_root)
    : DownloadDataset("DemoColoredICPPointClouds", data_descriptor, data_root) {
    paths_.push_back(GetExtractDir() + "/frag_115.ply");
    paths_.push_back(GetExtractDir() + "/frag_116.ply");
}

std::string DemoColoredICPPointClouds::GetPaths(size_t index) const {
    if (index > 1) {
        utility::LogError(
                "Invalid index. Expected index between 0 to 1 but got {}.",
                index);
    }
    return paths_[index];
}

}  // namespace data
}  // namespace open3d
