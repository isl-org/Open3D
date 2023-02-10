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
        Open3DDownloadsPrefix() + "20220301-data/DamagedHelmetModel.glb",
        "a3af6ad5a8329f22ba08b7f16e4a97d8"};

DamagedHelmetModel::DamagedHelmetModel(const std::string& data_root)
    : DownloadDataset("DamagedHelmetModel", data_descriptor, data_root) {
    path_ = GetExtractDir() + "/DamagedHelmetModel.glb";
}

}  // namespace data
}  // namespace open3d
