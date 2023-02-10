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
        Open3DDownloadsPrefix() + "20220301-data/point_cloud_sample1.pts",
        "5c2c618b703d0161e6e333fcbf55a1e9"};

PTSPointCloud::PTSPointCloud(const std::string& data_root)
    : DownloadDataset("PTSPointCloud", data_descriptor, data_root) {
    path_ = GetExtractDir() + "/point_cloud_sample1.pts";
}

}  // namespace data
}  // namespace open3d
