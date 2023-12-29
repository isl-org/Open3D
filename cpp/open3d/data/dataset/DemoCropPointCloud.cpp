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
        Open3DDownloadsPrefix() + "20220201-data/DemoCropPointCloud.zip",
        "12dbcdddd3f0865d8312929506135e23"};

DemoCropPointCloud::DemoCropPointCloud(const std::string& data_root)
    : DownloadDataset("DemoCropPointCloud", data_descriptor, data_root) {
    const std::string extract_dir = GetExtractDir();
    point_cloud_path_ = extract_dir + "/fragment.ply";
    cropped_json_path_ = extract_dir + "/cropped.json";
}

}  // namespace data
}  // namespace open3d
