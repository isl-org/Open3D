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
        Open3DDownloadsPrefix() + "20220201-data/fragment.ply",
        "831ecffd4d7cbbbe02494c5c351aa6e5"};

PLYPointCloud::PLYPointCloud(const std::string& data_root)
    : DownloadDataset("PLYPointCloud", data_descriptor, data_root) {
    path_ = GetExtractDir() + "/fragment.ply";
}

}  // namespace data
}  // namespace open3d
