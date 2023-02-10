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
        Open3DDownloadsPrefix() + "20220301-data/LoungeRGBDImages.zip",
        "cdd307caef898519a8829ce1b6ab9f75"};

LoungeRGBDImages::LoungeRGBDImages(const std::string& data_root)
    : DownloadDataset("LoungeRGBDImages", data_descriptor, data_root) {
    color_paths_.reserve(3000);
    depth_paths_.reserve(3000);
    const std::string extract_dir = GetExtractDir();
    const size_t n_zero = 6;
    for (int i = 1; i < 3000; ++i) {
        std::string idx = std::to_string(i);
        idx = std::string(n_zero - std::min(n_zero, idx.length()), '0') + idx;
        color_paths_.push_back(extract_dir + "/color/" + idx + ".png");
        depth_paths_.push_back(extract_dir + "/depth/" + idx + ".png");
    }

    trajectory_log_path_ = extract_dir + "/lounge_trajectory.log";
    reconstruction_path_ = extract_dir + "/lounge.ply";
}

}  // namespace data
}  // namespace open3d
