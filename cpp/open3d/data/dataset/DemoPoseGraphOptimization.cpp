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
        Open3DDownloadsPrefix() + "20220201-data/DemoPoseGraphOptimization.zip",
        "af085b28d79dea7f0a50aef50c96b62c"};

DemoPoseGraphOptimization::DemoPoseGraphOptimization(
        const std::string& data_root)
    : DownloadDataset("DemoPoseGraphOptimization", data_descriptor, data_root) {
    const std::string extract_dir = GetExtractDir();
    pose_graph_fragment_path_ =
            extract_dir + "/pose_graph_example_fragment.json";
    pose_graph_global_path_ = extract_dir + "/pose_graph_example_global.json";
}

}  // namespace data
}  // namespace open3d
