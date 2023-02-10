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

const static std::vector<DataDescriptor> data_descriptors = {
        {Open3DDownloadsPrefix() + "20220301-data/bedroom01.zip",
         "2d1018ceeb72680f5d16b2f419da9bb1"},
        {Open3DDownloadsPrefix() + "20220301-data/bedroom02.zip",
         "5e6ffbccc0907dc5acc374aa76a79081"},
        {Open3DDownloadsPrefix() + "20220301-data/bedroom03.zip",
         "ebf13b89ec364b1788dd492c27b9b800"},
        {Open3DDownloadsPrefix() + "20220301-data/bedroom04.zip",
         "94c0e6c862a54588582b06520946fb15"},
        {Open3DDownloadsPrefix() + "20220301-data/bedroom05.zip",
         "54b927edb6fd61838025bc66ed767408"},
};

BedroomRGBDImages::BedroomRGBDImages(const std::string& data_root)
    : DownloadDataset("BedroomRGBDImages", data_descriptors, data_root) {
    color_paths_.reserve(21931);
    depth_paths_.reserve(21931);
    const std::string extract_dir = GetExtractDir();
    const size_t n_zero = 6;
    for (int i = 1; i < 21931; ++i) {
        std::string idx = std::to_string(i);
        idx = std::string(n_zero - std::min(n_zero, idx.length()), '0') + idx;
        color_paths_.push_back(extract_dir + "/image/" + idx + ".jpg");
        depth_paths_.push_back(extract_dir + "/depth/" + idx + ".png");
    }

    trajectory_log_path_ = extract_dir + "/bedroom.log";
    reconstruction_path_ = extract_dir + "/bedroom.ply";
}

}  // namespace data
}  // namespace open3d
