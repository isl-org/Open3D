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
        Open3DDownloadsPrefix() + "20220201-data/EaglePointCloud.ply",
        "e4e6c77bc548e7eb7548542a0220ad78"};

EaglePointCloud::EaglePointCloud(const std::string& data_root)
    : DownloadDataset("EaglePointCloud", data_descriptor, data_root) {
    path_ = GetExtractDir() + "/EaglePointCloud.ply";
}

}  // namespace data
}  // namespace open3d
