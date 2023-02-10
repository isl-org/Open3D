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
        Open3DDownloadsPrefix() + "20220301-data/CrateModel.zip",
        "20413eada103969bb3ca5df9aebc2034"};

CrateModel::CrateModel(const std::string& data_root)
    : DownloadDataset("CrateModel", data_descriptor, data_root) {
    const std::string extract_dir = GetExtractDir();
    map_filename_to_path_ = {{"crate_material", extract_dir + "/crate.mtl"},
                             {"crate_model", extract_dir + "/crate.obj"},
                             {"texture_image", extract_dir + "/crate.jpg"}};
}

}  // namespace data
}  // namespace open3d
