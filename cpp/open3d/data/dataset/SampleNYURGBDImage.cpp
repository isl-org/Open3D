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
        Open3DDownloadsPrefix() + "20220201-data/SampleNYURGBDImage.zip",
        "b0baaf892c7ff9b202eb5fb40c5f7b58"};

SampleNYURGBDImage::SampleNYURGBDImage(const std::string& data_root)
    : DownloadDataset("SampleNYURGBDImage", data_descriptor, data_root) {
    color_path_ = GetExtractDir() + "/NYU_color.ppm";
    depth_path_ = GetExtractDir() + "/NYU_depth.pgm";
}

}  // namespace data
}  // namespace open3d
