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
        Open3DDownloadsPrefix() + "20220201-data/SampleSUNRGBDImage.zip",
        "b1a430586547c8986bdf8b36179a8e67"};

SampleSUNRGBDImage::SampleSUNRGBDImage(const std::string& data_root)
    : DownloadDataset("SampleSUNRGBDImage", data_descriptor, data_root) {
    color_path_ = GetExtractDir() + "/SUN_color.jpg";
    depth_path_ = GetExtractDir() + "/SUN_depth.png";
}

}  // namespace data
}  // namespace open3d
