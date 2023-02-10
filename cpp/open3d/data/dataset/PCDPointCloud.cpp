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
        Open3DDownloadsPrefix() + "20220201-data/fragment.pcd",
        "f3a613fd2bdecd699aabdd858fb29606"};

PCDPointCloud::PCDPointCloud(const std::string& data_root)
    : DownloadDataset("PCDPointCloud", data_descriptor, data_root) {
    path_ = GetExtractDir() + "/fragment.pcd";
}

}  // namespace data
}  // namespace open3d
