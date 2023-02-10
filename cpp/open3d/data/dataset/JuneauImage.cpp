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
        Open3DDownloadsPrefix() + "20220201-data/JuneauImage.jpg",
        "a090f6342893bdf0caefd83c6debbecd"};

JuneauImage::JuneauImage(const std::string& data_root)
    : DownloadDataset("JuneauImage", data_descriptor, data_root) {
    path_ = GetExtractDir() + "/JuneauImage.jpg";
}

}  // namespace data
}  // namespace open3d
