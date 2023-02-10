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
        Open3DDownloadsPrefix() + "20220301-data/WoodFloorTexture.zip",
        "f11b3e50208095e87340049b9ac3c319"};

WoodFloorTexture::WoodFloorTexture(const std::string& data_root)
    : DownloadDataset("WoodFloorTexture", data_descriptor, data_root) {
    const std::string extract_dir = GetExtractDir();
    map_filename_to_path_ = {
            {"albedo", extract_dir + "/WoodFloor050_Color.jpg"},
            {"normal", extract_dir + "/WoodFloor050_NormalDX.jpg"},
            {"roughness", extract_dir + "/WoodFloor050_Roughness.jpg"}};
}

}  // namespace data
}  // namespace open3d
