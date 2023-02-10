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
        Open3DDownloadsPrefix() + "redwood/office1-fragments-ply.zip",
        "c519fe0495b3c731ebe38ae3a227ac25"};

OfficePointClouds::OfficePointClouds(const std::string& data_root)
    : DownloadDataset("OfficePointClouds", data_descriptor, data_root) {
    paths_.reserve(53);
    for (int i = 0; i < 53; ++i) {
        paths_.push_back(GetExtractDir() + "/cloud_bin_" + std::to_string(i) +
                         ".ply");
    }
}

std::string OfficePointClouds::GetPaths(size_t index) const {
    if (index > 52) {
        utility::LogError(
                "Invalid index. Expected index between 0 to 52 but got {}.",
                index);
    }
    return paths_[index];
}

}  // namespace data
}  // namespace open3d
