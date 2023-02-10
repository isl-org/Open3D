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
        Open3DDownloadsPrefix() + "20220301-data/TerrazzoTexture.zip",
        "8d67f191fb5d80a27d8110902cac008e"};

TerrazzoTexture::TerrazzoTexture(const std::string& data_root)
    : DownloadDataset("TerrazzoTexture", data_descriptor, data_root) {
    const std::string extract_dir = GetExtractDir();
    map_filename_to_path_ = {
            {"albedo", extract_dir + "/Terrazzo018_Color.jpg"},
            {"normal", extract_dir + "/Terrazzo018_NormalDX.jpg"},
            {"roughness", extract_dir + "/Terrazzo018_Roughness.jpg"}};
}

}  // namespace data
}  // namespace open3d
