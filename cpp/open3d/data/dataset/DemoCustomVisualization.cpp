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
        Open3DDownloadsPrefix() + "20220301-data/DemoCustomVisualization.zip",
        "04cb716145c51d0119b59c7876249891"};

DemoCustomVisualization::DemoCustomVisualization(const std::string& data_root)
    : DownloadDataset("DemoCustomVisualization", data_descriptor, data_root) {
    const std::string extract_dir = GetExtractDir();
    point_cloud_path_ = extract_dir + "/fragment.ply";
    camera_trajectory_path_ = extract_dir + "/camera_trajectory.json";
    render_option_path_ = extract_dir + "/renderoption.json";
}

}  // namespace data
}  // namespace open3d
