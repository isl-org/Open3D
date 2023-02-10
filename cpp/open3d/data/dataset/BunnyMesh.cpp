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
        Open3DDownloadsPrefix() + "20220201-data/BunnyMesh.ply",
        "568f871d1a221ba6627569f1e6f9a3f2"};

BunnyMesh::BunnyMesh(const std::string& data_root)
    : DownloadDataset("BunnyMesh", data_descriptor, data_root) {
    path_ = GetExtractDir() + "/BunnyMesh.ply";
}

}  // namespace data
}  // namespace open3d
