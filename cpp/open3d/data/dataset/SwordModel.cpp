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
        Open3DDownloadsPrefix() + "20220301-data/SwordModel.zip",
        "eb7df358b5c31c839f03c4b3b4157c04"};

SwordModel::SwordModel(const std::string& data_root)
    : DownloadDataset("SwordModel", data_descriptor, data_root) {
    const std::string extract_dir = GetExtractDir();
    map_filename_to_path_ = {
            {"sword_material", extract_dir + "/UV.mtl"},
            {"sword_model", extract_dir + "/UV.obj"},
            {"base_color", extract_dir + "/UV_blinn1SG_BaseColor.png"},
            {"metallic", extract_dir + "/UV_blinn1SG_Metallic.png"},
            {"normal", extract_dir + "/UV_blinn1SG_Normal.png"},
            {"roughness", extract_dir + "/UV_blinn1SG_Roughness.png"}};
}

}  // namespace data
}  // namespace open3d
