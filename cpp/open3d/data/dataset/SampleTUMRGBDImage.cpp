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
        Open3DDownloadsPrefix() + "20220201-data/SampleTUMRGBDImage.zip",
        "91758d42b142dbad7b0d90e857ad47a8"};

SampleTUMRGBDImage::SampleTUMRGBDImage(const std::string& data_root)
    : DownloadDataset("SampleTUMRGBDImage", data_descriptor, data_root) {
    color_path_ = GetExtractDir() + "/TUM_color.png";
    depth_path_ = GetExtractDir() + "/TUM_depth.png";
}

}  // namespace data
}  // namespace open3d
