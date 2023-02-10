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
        Open3DDownloadsPrefix() + "20220201-data/ArmadilloMesh.ply",
        "9e68ff1b1cc914ed88cd84f6a8235021"};

ArmadilloMesh::ArmadilloMesh(const std::string& data_root)
    : DownloadDataset("ArmadilloMesh", data_descriptor, data_root) {
    path_ = GetExtractDir() + "/" + "ArmadilloMesh.ply";
}

}  // namespace data
}  // namespace open3d
