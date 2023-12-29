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
        Open3DDownloadsPrefix() + "20220201-data/KnotMesh.ply",
        "bfc9f132ecdfb7f9fdc42abf620170fc"};

KnotMesh::KnotMesh(const std::string& data_root)
    : DownloadDataset("KnotMesh", data_descriptor, data_root) {
    path_ = GetExtractDir() + "/KnotMesh.ply";
}

}  // namespace data
}  // namespace open3d
