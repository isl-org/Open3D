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
        Open3DDownloadsPrefix() + "xxx/SampleICPPointClouds.zip",
        "9d1ead73e678fa2f51a70a933b0bf017"};

SampleICPPointClouds::SampleICPPointClouds(const std::string& data_root)
    : DownloadDataset("SampleICPPointClouds", data_descriptor, data_root) {}

}  // namespace data
}  // namespace open3d
