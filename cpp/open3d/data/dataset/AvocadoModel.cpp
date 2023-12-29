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
        Open3DDownloadsPrefix() + "20220301-data/AvocadoModel.glb",
        "829f96a0a3a7d5556e0a263ea0699217"};

AvocadoModel::AvocadoModel(const std::string& data_root)
    : DownloadDataset("AvocadoModel", data_descriptor, data_root) {
    path_ = GetExtractDir() + "/AvocadoModel.glb";
}

}  // namespace data
}  // namespace open3d
